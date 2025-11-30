import asyncio
import logging
import os
import sqlite3
import time
import re
import configparser

from pathlib import Path
from typing import List, Optional
from base64 import b64decode, b64encode
from io import BytesIO

from openai import OpenAI

from telegram import Update, BotCommand
from telegram.constants import ChatType, ChatAction, MessageEntityType, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

BOT_CONF_PATH = os.getenv("BOT_CONF_PATH", "bot.conf")

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Tone down noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- OpenAI client + config ---
client = OpenAI()  # uses OPENAI_API_KEY env var
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant chatting through a Telegram bot. "
    "Be concise but clear. When you output code or configuration, "
    "format it using fenced code blocks like:\n"
    "```language\n"
    "code here\n"
    "```\n"
    "Use standard Markdown and avoid unclosed backticks."
)

# --- SQLite config ---
BOT_DB_PATH = os.getenv("BOT_DB_PATH", "state/conversations.db")

def parse_id_list(value: str | None) -> set[int]:
    if not value:
        return set()
    ids: set[int] = set()
    for token in value.replace(",", " ").split():
        try:
            ids.add(int(token))
        except ValueError:
            logger.warning("Ignoring non-numeric id token %r in whitelist", token)
    return ids

def load_whitelists_from_conf() -> tuple[set[int], set[int]]:
    users: set[int] = set()
    groups: set[int] = set()

    cfg = configparser.ConfigParser()
    read_files = cfg.read(BOT_CONF_PATH, encoding="utf-8")
    if not read_files:
        logger.error("Whitelist config file %s not found or unreadable", BOT_CONF_PATH)
        return users, groups

    if "whitelist" not in cfg:
        logger.error("Config %s missing [whitelist] section", BOT_CONF_PATH)
        return users, groups

    sec = cfg["whitelist"]
    users = parse_id_list(sec.get("users"))
    groups = parse_id_list(sec.get("groups"))

    logger.info("Loaded %d allowed users and %d allowed groups", len(users), len(groups))
    return users, groups


# --- Whitelists ---
ALLOWED_USERS: set[int] = set()
ALLOWED_GROUPS: set[int] = set()

RECENT_MESSAGES_FOR_CONTEXT = 30  # K
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", OPENAI_MODEL)
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_PROMPT_MODEL = os.getenv("OPENAI_IMAGE_PROMPT_MODEL", OPENAI_MODEL)
OPENAI_PROFILE_MODEL = os.getenv("OPENAI_PROFILE_MODEL", OPENAI_MODEL)
PROFILE_UPDATE_EVERY = 10  # how often to update user profile from summary
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", OPENAI_MODEL)

IMAGE_PROMPT_SYSTEM = (
    "You are an assistant that rewrites user requests into detailed prompts for an "
    "AI image generation model.\n\n"
    "Given a short user description, produce a single concise English prompt that "
    "fully specifies:\n"
    "- the main subject(s)\n"
    "- important attributes (pose, clothing, expression, etc.)\n"
    "- environment / background\n"
    "- mood / lighting\n"
    "- style (e.g., digital painting, pixel art, watercolor, 3D render, etc.) if implied\n\n"
    "Rules:\n"
    "- Do NOT mention 'in this image' or 'an image of'; just describe the scene directly.\n"
    "- Do NOT include markdown, quotes, or bullet points.\n"
    "- One sentence or two short sentences max.\n"
)

USER_PROFILE_SYSTEM = (
    "You maintain a persistent profile of the user based on their messages.\n\n"
    "The profile is a bullet list of relatively stable facts about the user: "
    "background, preferences, projects, expertise, language preferences, etc.\n\n"
    "Rules:\n"
    "- Only include information that seems likely to stay true for months or years.\n"
    "- Ignore ephemeral details (like what they ate today).\n"
    "- Never include passwords, tokens, secrets, or any sensitive security data.\n"
    "- Keep the profile concise but specific, 5–20 bullets.\n"
    "- If something in the existing profile is clearly contradicted, update it.\n"
)


def get_user_profile_text(user_id: int) -> str:
    """
    Return the stored profile text for this user, or empty string if none.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT profile FROM user_profiles WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row["profile"] if row else ""


def set_user_profile_text(user_id: int, profile: str) -> None:
    now = int(time.time())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_profiles (user_id, profile, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET profile = excluded.profile,
                      updated_at = excluded.updated_at
        """,
        (user_id, profile, now),
    )
    conn.commit()
    conn.close()


async def update_user_profile_from_summary(user_id: int, summary_text: str) -> None:
    text = summary_text.strip()
    if len(text) < 40:
        return

    existing_profile = get_user_profile_text(user_id)

    def _call_profile_model() -> str:
        user_content = (
            "Existing profile (may be empty):\n"
            f"{existing_profile or '(none)'}\n\n"
            "Conversation summary:\n"
            f"{text}\n\n"
            "Update the profile, returning ONLY the new full bullet list focused on this user. "
            "Ignore details about other participants or the assistant."
        )

        resp = client.chat.completions.create(
            model=OPENAI_PROFILE_MODEL,
            messages=[
                {"role": "system", "content": USER_PROFILE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        new_profile = await asyncio.to_thread(_call_profile_model)
        if new_profile:
            set_user_profile_text(user_id, new_profile)
    except Exception:
        logger.exception("Failed to update user profile from summary for user_id=%s", user_id)


async def build_image_prompt(raw_prompt: str) -> str:
    """Use a chat model to turn a rough user request into a polished image prompt."""
    raw_prompt = raw_prompt.strip()
    if not raw_prompt:
        return raw_prompt

    def _call_prompt_llm() -> str:
        resp = client.chat.completions.create(
            model=OPENAI_IMAGE_PROMPT_MODEL,
            messages=[
                {"role": "system", "content": IMAGE_PROMPT_SYSTEM},
                {"role": "user", "content": raw_prompt},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        improved = await asyncio.to_thread(_call_prompt_llm)
        if improved:
            return improved
    except Exception:
        logger.exception("Image prompt generation failed")

    # fallback
    return raw_prompt


# ========== DB helpers ==========

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        BOT_DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    Path(BOT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection()
    cur = conn.cursor()

    # Conversations
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id      INTEGER NOT NULL,
            thread_id    INTEGER,
            created_at   INTEGER NOT NULL,
            last_used_at INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_chat_thread
        ON conversations(chat_id, thread_id);
        """
    )

    # Messages
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,       -- 'system' | 'user' | 'assistant'
            content         TEXT NOT NULL,
            created_at      INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_msg_conv_created
        ON messages(conversation_id, created_at);
        """
    )

    # Topics: per chat/topic system prompt and metadata
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS topics (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id       INTEGER NOT NULL,
            thread_id     INTEGER,
            system_prompt TEXT,
            created_at    INTEGER NOT NULL,
            updated_at    INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_chat_thread
        ON topics(chat_id, thread_id);
        """
    )
    # Conversation summaries: one per conversation, covers all messages up to last_message_id
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            conversation_id INTEGER PRIMARY KEY
                REFERENCES conversations(id) ON DELETE CASCADE,
            summary         TEXT NOT NULL,
            last_message_id INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );
        """
    )
    # Persistent per-user profiles (long-lived facts/preferences)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id    INTEGER PRIMARY KEY,
            profile    TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", BOT_DB_PATH)


def get_or_create_conversation(chat_id: int, thread_id: Optional[int]) -> int:
    now = int(time.time())
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM conversations WHERE chat_id = ? AND thread_id IS ?",
        (chat_id, thread_id),
    )
    row = cur.fetchone()

    if row:
        conv_id = row["id"]
        cur.execute(
            "UPDATE conversations SET last_used_at = ? WHERE id = ?",
            (now, conv_id),
        )
    else:
        cur.execute(
            """
            INSERT INTO conversations (chat_id, thread_id, created_at, last_used_at)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, thread_id, now, now),
        )
        conv_id = cur.lastrowid

    conn.commit()
    conn.close()
    return conv_id


def add_message_db(conversation_id: int, role: str, content: str) -> None:
    now = int(time.time())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO messages (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, role, content, now),
    )
    cur.execute(
        "UPDATE conversations SET last_used_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    conn.close()


def get_recent_messages_db(conversation_id: int, limit: int) -> List[dict]:
    """
    Return last `limit` messages as list of {role, content}, in chronological order.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (conversation_id, limit),
    )
    rows = cur.fetchall()
    conn.close()

    msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    return msgs


def delete_conversation(chat_id: int, thread_id: Optional[int]) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM conversations WHERE chat_id = ? AND thread_id IS ?",
        (chat_id, thread_id),
    )
    conn.commit()
    conn.close()


# --- markdown helpers ---

MDV2_SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!\\"

def escape_markdown_v2_text(text: str) -> str:
    """
    Escape Telegram MarkdownV2 special characters in non-code text.
    """
    return re.sub(
        fr"([{re.escape(MDV2_SPECIAL_CHARS)}])",
        r"\\\1",
        text,
    )

def format_for_telegram_markdown_v2(text: str) -> str:
    """
    Escape MarkdownV2 special chars everywhere EXCEPT inside fenced code blocks ```...```.

    This assumes ChatGPT-style triple-backtick fences.
    """
    # Split on fenced code blocks, keep the fences
    parts = re.split(r"(```[\s\S]*?```)", text)

    formatted_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```"):
            # Leave fenced code blocks as-is
            formatted_parts.append(part)
        else:
            # Escape normal text
            formatted_parts.append(escape_markdown_v2_text(part))

    return "".join(formatted_parts)


# --- Topic prompt helpers ---

def get_topic_prompt_db(chat_id: int, thread_id: Optional[int]) -> Optional[str]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT system_prompt FROM topics WHERE chat_id = ? AND thread_id IS ?",
        (chat_id, thread_id),
    )
    row = cur.fetchone()
    conn.close()
    if row and row["system_prompt"]:
        return row["system_prompt"]
    return None


def set_topic_prompt_db(chat_id: int, thread_id: Optional[int], prompt: str) -> None:
    now = int(time.time())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO topics (chat_id, thread_id, system_prompt, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(chat_id, thread_id)
        DO UPDATE SET system_prompt = excluded.system_prompt,
                      updated_at    = excluded.updated_at
        """,
        (chat_id, thread_id, prompt, now, now),
    )
    conn.commit()
    conn.close()


def clear_topic_prompt_db(chat_id: int, thread_id: Optional[int]) -> None:
    now = int(time.time())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE topics
        SET system_prompt = NULL, updated_at = ?
        WHERE chat_id = ? AND thread_id IS ?
        """,
        (now, chat_id, thread_id),
    )
    conn.commit()
    conn.close()


def extract_image_prompt_from_text(text: str) -> Optional[str]:
    """
    If the text looks like an image request (starts with 'draw' or 'нарисуй',
    in ANY case), return the rest of the text as the raw prompt.
    Otherwise return None.

    Examples:
      'draw a cat on a bike'      -> 'a cat on a bike'
      'Draw A Cat On A Bike'      -> 'A Cat On A Bike'
      'НАРИСУЙ лягушку'          -> 'лягушку'
    """
    t = text.strip()
    if not t:
        return None

    low = t.casefold()
    keywords = ("draw", "нарисуй")

    for kw in keywords:
        kw_low = kw.casefold()

        # Exact match: "draw" / "DRAW" / "НАРИСУЙ" etc.
        if low == kw_low:
            return ""

        # Starts with keyword + space: "draw ..." / "DRAW ..." / "нарисуй ..."
        if low.startswith(kw_low + " "):
            # Strip off the keyword from the ORIGINAL text so we preserve casing
            return t[len(kw):].lstrip(" \t:-,.;")

    return None


# ========== whitelist helpers ==========

def is_authorized_user(update: Update) -> bool:
    user = update.effective_user
    if user is None:
        return False

    if not ALLOWED_USERS:
        logger.warning(
            "ALLOWED_USERS is empty; denying user id=%s username=%s",
            user.id,
            getattr(user, "username", None),
        )
        return False

    if user.id not in ALLOWED_USERS:
        logger.info(
            "Unauthorized USER: id=%s username=%s",
            user.id,
            getattr(user, "username", None),
        )
        return False

    return True


async def ensure_chat_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    if chat is None:
        return False

    if chat.type == ChatType.PRIVATE:
        return True

    if not ALLOWED_GROUPS:
        logger.warning(
            "ALLOWED_GROUPS is empty; denying chat id=%s title=%r",
            chat.id,
            getattr(chat, "title", None),
        )
        return False

    if chat.id not in ALLOWED_GROUPS:
        logger.info(
            "Unauthorized CHAT: id=%s type=%s title=%r",
            chat.id,
            chat.type,
            getattr(chat, "title", None),
        )
        return False

    return True


# ========== conversation key & reply rules ==========

def get_conversation_key(update: Update) -> tuple[int, Optional[int]]:
    chat = update.effective_chat
    msg = update.message

    if chat is None:
        return (0, None)

    thread_id: Optional[int] = None
    if msg is not None and msg.is_topic_message:
        thread_id = msg.message_thread_id

    return (chat.id, thread_id)


def should_respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    1. Do NOT respond if the message is a reply to someone else's message
       (i.e. not the bot and not the same user).
    2. Do NOT respond if someone is mentioned and the bot is NOT.

    Works for:
      - plain text messages (msg.text + msg.entities)
      - media captions (msg.caption + msg.caption_entities)
    """
    msg = update.message
    if msg is None:
        return False

    bot = context.bot
    bot_id = bot.id
    bot_username = bot.username

    # --- Rule 2: replies ---
    if msg.reply_to_message:
        replied = msg.reply_to_message
        replied_user = replied.from_user

        if replied_user is None:
            return False

        # Replies to the bot: always OK
        if replied_user.id == bot_id:
            return True

        # Replies to self: allow
        if msg.from_user and replied_user.id == msg.from_user.id:
            return True

        # Replies to any other user/bot -> ignore
        return False

    # Decide which text/entities to inspect: text or caption
    if msg.text:
        text = msg.text
        entities = msg.entities or []
    elif msg.caption:
        text = msg.caption
        entities = msg.caption_entities or []
    else:
        # No textual content: no mentions => OK
        return True

    # --- Rule 1: mentions in text / caption ---
    mentioned_usernames = set()
    mentioned_user_ids = set()

    for ent in entities:
        if ent.type == MessageEntityType.MENTION:
            segment = text[ent.offset : ent.offset + ent.length]
            if segment.startswith("@"):
                mentioned_usernames.add(segment[1:])
        elif ent.type == MessageEntityType.TEXT_MENTION and ent.user:
            mentioned_user_ids.add(ent.user.id)

    mentions_anyone = bool(mentioned_usernames or mentioned_user_ids)

    # Did they mention the bot?
    mentions_bot = False
    if bot_username:
        mentions_bot = any(
            name.casefold() == bot_username.casefold()
            for name in mentioned_usernames
        )
    if not mentions_bot and bot_id:
        mentions_bot = bot_id in mentioned_user_ids

    # If someone is mentioned and it's not the bot -> don't respond
    if mentions_anyone and not mentions_bot:
        return False

    return True


def get_summary_record(conversation_id: int) -> tuple[Optional[str], int]:
    """
    Return (summary_text, last_message_id).
    If no summary exists yet, returns (None, 0).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT summary, last_message_id FROM conversation_summaries WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None, 0
    return row["summary"], row["last_message_id"]


def set_summary_record(conversation_id: int, summary: str, last_message_id: int) -> None:
    now = int(time.time())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO conversation_summaries (conversation_id, summary, last_message_id, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(conversation_id)
        DO UPDATE SET summary = excluded.summary,
                      last_message_id = excluded.last_message_id,
                      updated_at      = excluded.updated_at
        """,
        (conversation_id, summary, last_message_id, now),
    )
    conn.commit()
    conn.close()


def get_max_message_id(conversation_id: int) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT MAX(id) AS max_id FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row["max_id"] or 0


async def ensure_summary_covers_old(conversation_id: int) -> Optional[str]:
    """
    Ensure that all messages older than the last K messages are covered by the summary.

    Invariant after this function:
        let n = max message id, K = RECENT_MESSAGES_FOR_CONTEXT
        if n <= K: summary may be None
        if n >  K: summary exists and covers all messages with id <= n-K

    Returns the current summary text (possibly None if conversation is very short).
    """
    K = RECENT_MESSAGES_FOR_CONTEXT

    n = get_max_message_id(conversation_id)
    if n == 0:
        return None

    # If everything fits in the window, no summary needed
    if n <= K:
        summary_text, _ = get_summary_record(conversation_id)
        return summary_text

    boundary = n - K   # everything <= boundary must be summarized

    existing_summary, last_id = get_summary_record(conversation_id)
    if last_id >= boundary:
        # Summary already covers everything older than the window
        return existing_summary

    # We must extend the summary to cover at least up to 'boundary'.
    start_id = last_id + 1  # first unsummarized message
    #end_id = n              # we choose to summarize up to latest (can be boundary if you prefer)
    end_id = boundary

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, role, content
        FROM messages
        WHERE conversation_id = ?
          AND id >= ?
          AND id <= ?
        ORDER BY id ASC
        """,
        (conversation_id, start_id, end_id),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        # This shouldn't normally happen but be defensive
        return existing_summary

    # Build transcript of the unsummarized range
    lines = []
    for r in rows:
        role = r["role"]
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "System"
        lines.append(f"{prefix}: {r['content']}")
    transcript = "\n".join(lines)

    if not transcript.strip():
        return existing_summary

    # Build summarization prompt
    if existing_summary:
        user_content = (
            "You are updating an existing summary of a conversation.\n\n"
            "Existing summary:\n"
            f"{existing_summary}\n\n"
            "Here is additional conversation transcript. "
            "Refine and update the summary so it reflects the entire conversation so far.\n\n"
            f"{transcript}"
        )
    else:
        user_content = (
            "You are summarizing a conversation.\n\n"
            "Here is the conversation transcript:\n"
            f"{transcript}\n\n"
            "Write a concise summary capturing key facts, decisions, tasks, and user preferences."
        )

    def _call_summarizer() -> str:
        resp = client.chat.completions.create(
            model=OPENAI_SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You create concise summaries of conversations. "
                        "Reply ONLY with the summary text."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    try:
        new_summary = await asyncio.to_thread(_call_summarizer)
    except Exception:
        logger.exception("Failed to summarize conversation %s", conversation_id)
        # In worst case we fall back to existing summary and accept that the gap persists
        return existing_summary

    # We summarized everything up to 'end_id' (>= boundary),
    # so after this call: last_message_id == end_id >= n-K, no gap.
    set_summary_record(conversation_id, new_summary, end_id)
    logger.info(
        "Updated summary for conversation %s up to message id %s (n=%s, boundary=%s)",
        conversation_id, end_id, n, boundary,
    )
    return new_summary


# ========== Handlers ==========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return

    if update.message:
        await update.message.reply_text(
            "Hi! I'm a ChatGPT-style bot with per-topic memory.\n"
            "Send me a message and I'll answer.\n"
            "Use /new to reset this chat/topic.\n"
            "Use /topic_prompt <text> to set a custom persona for this chat/topic."
        )


async def handle_image_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    raw_user_text: str,
    image_request_text: str,
) -> None:
    msg = update.message
    chat = update.effective_chat
    if msg is None or chat is None:
        return

    thread_id = msg.message_thread_id if msg.is_topic_message else None

    conv_id = get_or_create_conversation(chat.id, thread_id)
    add_message_db(conv_id, "user", raw_user_text)

    if not image_request_text and msg.reply_to_message and msg.reply_to_message.text:
        image_request_text = msg.reply_to_message.text.strip()

    if not image_request_text:
        await msg.reply_text(
            "You said 'draw', but I don't see what to draw.\n"
            "Example: 'draw a frog on a skateboard'."
        )
        return

    improved_prompt = await build_image_prompt(image_request_text)

    # Quick status message
    status = await msg.reply_text("Got your request, generating an image…")

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.UPLOAD_PHOTO)

    try:
        def _call_image_api():
            return client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=improved_prompt,
                size="1024x1024",
                n=1,
            )

        resp = await asyncio.to_thread(_call_image_api)
        image_b64 = resp.data[0].b64_json
        image_bytes = b64decode(image_b64)
    except Exception:
        logger.exception("OpenAI image API error")
        try:
            await context.bot.edit_message_text(
                chat_id=status.chat_id,
                message_id=status.message_id,
                text="Error generating image. Please try again later.",
            )
        except Exception:
            await msg.reply_text("Error generating image. Please try again later.")
        return

    caption = f"🖼️ Image generated. Prompt: {improved_prompt}"
    add_message_db(conv_id, "assistant", caption)

    try:
        await context.bot.delete_message(
            chat_id=status.chat_id,
            message_id=status.message_id,
        )
    except Exception:
        pass

    try:
        await msg.reply_photo(image_bytes, caption=caption)
    except Exception:
        logger.exception("Failed to send generated image")
        await msg.reply_text("Generated an image but failed to send it via Telegram.")


async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /img <prompt>
    /image <prompt>

    Or reply /img to a message to use that message text as the prompt.
    """
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return

    msg = update.message
    if msg is None:
        return

    # 1) Get raw prompt: args after command OR replied message text
    prompt_parts = context.args
    raw_prompt = " ".join(prompt_parts).strip()

    if not raw_prompt and msg.reply_to_message and msg.reply_to_message.text:
        raw_prompt = msg.reply_to_message.text.strip()

    if not raw_prompt:
        await msg.reply_text("Usage: /img <description> or reply /img to a message.")
        return

    # 2) Use chat model to turn raw_prompt into a better image prompt
    improved_prompt = await build_image_prompt(raw_prompt)

    # Send a quick status message
    status = await msg.reply_text("Got your request, generating an image…")

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.UPLOAD_PHOTO)

    try:
        def _call_image_api():
            return client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=improved_prompt,
                size="1024x1024",
                n=1,
            )

        resp = await asyncio.to_thread(_call_image_api)
        image_b64 = resp.data[0].b64_json
        image_bytes = b64decode(image_b64)
    except Exception:
        logger.exception("OpenAI image API error")
        # Turn the status into an error message
        try:
            await context.bot.edit_message_text(
                chat_id=status.chat_id,
                message_id=status.message_id,
                text="Error generating image. Please try again later.",
            )
        except Exception:
            await msg.reply_text("Error generating image. Please try again later.")
        return

    # Success: remove the status and send the image
    try:
        await context.bot.delete_message(
            chat_id=status.chat_id,
            message_id=status.message_id,
        )
    except Exception:
        # If we can't delete, it's not fatal; we'll just leave it.
        pass

    try:
        caption = f"Prompt: {improved_prompt}"
        await msg.reply_photo(image_bytes, caption=caption)
    except Exception:
        logger.exception("Failed to send generated image")
        await msg.reply_text("Generated an image but failed to send it via Telegram.")


async def new_conv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return
    if update.message is None:
        return

    chat = update.effective_chat
    msg = update.message
    if chat is None:
        return

    thread_id = msg.message_thread_id if msg.is_topic_message else None
    delete_conversation(chat.id, thread_id)
    await update.message.reply_text("Started a new conversation in this chat/topic.")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming photos: send them to a vision-capable OpenAI model
    together with the caption (if any), and reply with the analysis.
    """
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return
    if not should_respond(update, context):
        return

    msg = update.message
    chat = update.effective_chat
    user = update.effective_user

    if msg is None or chat is None or not msg.photo:
        return

    # Use the highest-resolution photo Telegram gives us
    tg_photo = msg.photo[-1]

    try:
        file = await context.bot.get_file(tg_photo.file_id)
    except Exception:
        logger.exception("Failed to get Telegram file for photo")
        return

    # Download the image into memory
    bio = BytesIO()
    try:
        await file.download_to_memory(out=bio)
    except Exception:
        logger.exception("Failed to download photo to memory")
        return

    image_bytes = bio.getvalue()
    if not image_bytes:
        await msg.reply_text("Couldn't download the image data.")
        return

    # Encode as data URL for OpenAI vision
    image_b64 = b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{image_b64}"

    caption = (msg.caption or "").strip()

    # Conversation identification (chat + optional topic thread)
    thread_id = msg.message_thread_id if msg.is_topic_message else None
    conv_id = get_or_create_conversation(chat.id, thread_id)

    # Store a textual representation of the photo in the DB
    if caption:
        stored_text = f"[Photo] {caption}"
    else:
        stored_text = "[Photo]"
    add_message_db(conv_id, "user", stored_text)

    # Maintain zero-gap summary
    summary_text = await ensure_summary_covers_old(conv_id)

    # Topic-specific system prompt (if any)
    topic_prompt_text = get_topic_prompt_db(chat.id, thread_id)
    system_prompt = topic_prompt_text or DEFAULT_SYSTEM_PROMPT

    # Build text history: system + summary + last K messages
    recent = get_recent_messages_db(conv_id, RECENT_MESSAGES_FOR_CONTEXT)

    history: List[dict] = [{"role": "system", "content": system_prompt}]
    if summary_text:
        history.append(
            {
                "role": "system",
                "content": f"Summary of earlier conversation so far:\n{summary_text}",
            }
        )
    for m in recent:
        history.append(
            {
                "role": m["role"],
                "content": m["content"],
            }
        )

    # Add the multimodal user message with the actual image
    if caption:
        text_for_image = caption
    else:
        text_for_image = "Describe this image in detail."

    history.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_for_image},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    )

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_VISION_MODEL,
            messages=history,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
    except Exception:
        logger.exception("OpenAI vision API error")
        await msg.reply_text("Error analyzing the image. Please try again later.")
        return

    # Store assistant reply in the DB and send it back
    add_message_db(conv_id, "assistant", answer)
    await msg.reply_text(answer)


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle uploaded documents (e.g., PDFs, text files):
    - Download from Telegram
    - Extract text (PDF via PyPDF2, text files as-is)
    - Pass the text + caption to the chat model with full conversation context
    """
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return
    if not should_respond(update, context):
        return

    msg = update.message
    chat = update.effective_chat
    user = update.effective_user

    if msg is None or chat is None or msg.document is None:
        return

    doc = msg.document
    file_name = doc.file_name or "document"
    mime = doc.mime_type or ""

    # Decide what we can handle
    is_pdf = mime == "application/pdf" or file_name.lower().endswith(".pdf")
    is_text = mime.startswith("text/") or file_name.lower().endswith(".txt")

    if not (is_pdf or is_text):
        await msg.reply_text(
            f"Document type not supported yet ({mime or 'unknown'}). "
            "Currently I handle PDFs and plain text files."
        )
        return

    # Download the file into memory
    try:
        tg_file = await context.bot.get_file(doc.file_id)
    except Exception:
        logger.exception("Failed to get Telegram file for document")
        await msg.reply_text("Couldn't download the document.")
        return

    bio = BytesIO()
    try:
        await tg_file.download_to_memory(out=bio)
        data = bio.getvalue()
    except Exception:
        logger.exception("Failed to download document to memory")
        await msg.reply_text("Couldn't download the document.")
        return

    if not data:
        await msg.reply_text("Downloaded document is empty.")
        return

    # Extract text
    extracted_text: str = ""
    if is_text:
        try:
            extracted_text = data.decode("utf-8", errors="replace")
        except Exception:
            logger.exception("Failed to decode text document")
            await msg.reply_text("Couldn't decode the text document.")
            return
    elif is_pdf:
        if PyPDF2 is None:
            await msg.reply_text(
                "PDF support is not enabled on the server (PyPDF2 not installed)."
            )
            return
        try:
            reader = PyPDF2.PdfReader(BytesIO(data))
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            extracted_text = "\n\n".join(pages)
        except Exception:
            logger.exception("Failed to extract text from PDF")
            await msg.reply_text("Couldn't extract text from the PDF.")
            return

    extracted_text = (extracted_text or "").strip()
    if not extracted_text:
        await msg.reply_text("The document appears to have no extractable text.")
        return

    # Build a short marker for the DB (we DON'T store full doc text there)
    caption = (msg.caption or "").strip()
    if caption:
        marker = f"[Document: {file_name}] {caption}"
    else:
        marker = f"[Document: {file_name}]"

    thread_id = msg.message_thread_id if msg.is_topic_message else None
    conv_id = get_or_create_conversation(chat.id, thread_id)

    # Store just the marker as the user message
    add_message_db(conv_id, "user", marker)

    # Maintain zero-gap summary
    summary_text = await ensure_summary_covers_old(conv_id)

    # Build system prompt with topic persona + user profile
    topic_prompt_text = get_topic_prompt_db(chat.id, thread_id)
    base_system_prompt = topic_prompt_text or DEFAULT_SYSTEM_PROMPT

    profile_text = get_user_profile_text(user.id) if user is not None else ""
    if profile_text:
        system_prompt = (
            base_system_prompt
            + "\n\nThe following is known about this user from previous conversations:\n"
            + profile_text
        )
    else:
        system_prompt = base_system_prompt

    # Recent context from DB
    recent = get_recent_messages_db(conv_id, RECENT_MESSAGES_FOR_CONTEXT)

    history: List[dict] = [{"role": "system", "content": system_prompt}]
    if summary_text:
        history.append(
            {
                "role": "system",
                "content": f"Summary of earlier conversation so far:\n{summary_text}",
            }
        )
    for m in recent:
        history.append(
            {
                "role": m["role"],
                "content": m["content"],
            }
        )

    # Truncate document text so we don't blow the context window
    MAX_DOC_CHARS = 8000
    doc_text = extracted_text
    truncated_note = ""
    if len(doc_text) > MAX_DOC_CHARS:
        truncated_note = (
            f"(Document text truncated to first {MAX_DOC_CHARS} characters "
            f"out of {len(doc_text)}.)\n\n"
        )
        doc_text = doc_text[:MAX_DOC_CHARS]

    if caption:
        user_instruction = caption
    else:
        user_instruction = "Please summarize this document and highlight the key points."

    user_content = (
        f"The user uploaded a document named '{file_name}'.\n"
        f"{truncated_note}"
        f"Document text:\n{doc_text}\n\n"
        f"User request: {user_instruction}"
    )

    history.append({"role": "user", "content": user_content})

    # Let the user know we're working on it
    status = await msg.reply_text("Got your document, analyzing it…")
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    # Call OpenAI (non-streaming here for simplicity)
    try:
        def _call_openai() -> str:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=history,
                temperature=0.4,
            )
            return resp.choices[0].message.content or ""

        answer = await asyncio.to_thread(_call_openai)
    except Exception:
        logger.exception("OpenAI API error while analyzing document")
        try:
            await context.bot.edit_message_text(
                chat_id=status.chat_id,
                message_id=status.message_id,
                text="Error analyzing the document. Please try again later.",
            )
        except Exception:
            await msg.reply_text("Error analyzing the document. Please try again later.")
        return

    # Store assistant reply in DB
    add_message_db(conv_id, "assistant", answer)

    # Format with MarkdownV2 (so code blocks etc. render if present)
    formatted = format_for_telegram_markdown_v2(answer)

    # Replace the status message with the final answer
    try:
        await context.bot.edit_message_text(
            chat_id=status.chat_id,
            message_id=status.message_id,
            text=formatted,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception:
        await msg.reply_text(formatted, parse_mode=ParseMode.MARKDOWN_V2)


async def topic_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /topic_prompt            -> show current prompt (or default)
    /topic_prompt <new text> -> set prompt for this chat/topic
    """
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return
    if update.message is None:
        return

    chat = update.effective_chat
    msg = update.message
    if chat is None:
        return

    thread_id = msg.message_thread_id if msg.is_topic_message else None
    args = context.args

    if not args:
        prompt = get_topic_prompt_db(chat.id, thread_id)
        if prompt:
            await msg.reply_text(
                "Current custom system prompt for this chat/topic:\n\n"
                f"{prompt}"
            )
        else:
            await msg.reply_text(
                "This chat/topic is currently using the default system prompt."
            )
        return

    new_prompt = " ".join(args).strip()
    if not new_prompt:
        await msg.reply_text("Prompt is empty, nothing changed.")
        return

    set_topic_prompt_db(chat.id, thread_id, new_prompt)
    await msg.reply_text(
        "Updated system prompt for this chat/topic.\n\n"
        f"New prompt:\n{new_prompt}"
    )


async def topic_prompt_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /topic_prompt_reset -> clear custom prompt, fall back to default for this chat/topic
    """
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return
    if update.message is None:
        return

    chat = update.effective_chat
    msg = update.message
    if chat is None:
        return

    thread_id = msg.message_thread_id if msg.is_topic_message else None
    clear_topic_prompt_db(chat.id, thread_id)
    await msg.reply_text(
        "Cleared custom system prompt for this chat/topic. "
        "Default prompt will be used from now on."
    )


async def chatgpt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        return
    if not await ensure_chat_allowed(update, context):
        return
    if not should_respond(update, context):
        return

    msg = update.message
    if msg is None or msg.text is None:
        return

    user_text = msg.text.strip()
    if not user_text:
        return

    user = update.effective_user

    # --- Image request branch: freeform "draw/нарисуй ..." ---
    img_raw = extract_image_prompt_from_text(user_text)
    if img_raw is not None:
        await handle_image_request(
            update,
            context,
            raw_user_text=user_text,
            image_request_text=img_raw,
        )
        return

    chat = update.effective_chat
    if chat is None:
        return

    thread_id = msg.message_thread_id if msg.is_topic_message else None

    # Conversation row
    conv_id = get_or_create_conversation(chat.id, thread_id)

    # Store user message in conversation history
    add_message_db(conv_id, "user", user_text)

    # Ensure summary has no gap w.r.t. the recent window
    summary_text = await ensure_summary_covers_old(conv_id)

    # --- Option B: update user profile from CONVERSATION SUMMARY occasionally ---
    if user is not None and summary_text:
        # e.g. once every N messages in this conversation
        n = get_max_message_id(conv_id)
        if n % PROFILE_UPDATE_EVERY == 0:
            # Run profile update in background; do not block main reply
            asyncio.create_task(
                update_user_profile_from_summary(user.id, summary_text)
            )

    # System prompt: topic-specific or default
    topic_prompt_text = get_topic_prompt_db(chat.id, thread_id)
    base_system_prompt = topic_prompt_text or DEFAULT_SYSTEM_PROMPT

    # Inject long-lived user profile (shared across topics/chats)
    profile_text = get_user_profile_text(user.id) if user is not None else ""
    if profile_text:
        system_prompt = (
            base_system_prompt
            + "\n\nThe following is known about this user from previous conversations:\n"
            + profile_text
        )
    else:
        system_prompt = base_system_prompt

    # Build context: system + summary + last K raw messages
    recent = get_recent_messages_db(conv_id, RECENT_MESSAGES_FOR_CONTEXT)

    history: List[dict] = [{"role": "system", "content": system_prompt}]
    if summary_text:
        history.append(
            {
                "role": "system",
                "content": f"Summary of earlier conversation so far:\n{summary_text}",
            }
        )
    history.extend(recent)

    # --- STREAMING PART STARTS HERE ---

    # Show typing and send a placeholder message we can edit
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)
    placeholder = await msg.reply_text("…")  # initial stub

    loop = asyncio.get_running_loop()
    chat_id = placeholder.chat_id
    message_id = placeholder.message_id

    def _stream_openai() -> str:
        """
        Run the OpenAI streaming call in a worker thread and
        update the Telegram message from that thread.
        Returns the full final answer text.
        """
        answer_parts: list[str] = []
        last_edit = 0.0

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            temperature=0.7,
            stream=True,
        )

        for chunk in stream:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            piece = delta.content or ""
            if not piece:
                continue

            answer_parts.append(piece)
            full_text = "".join(answer_parts)

            # Throttle edits a bit so we don't spam Telegram
            now = time.time()
            if now - last_edit > 0.2:
                last_edit = now
                asyncio.run_coroutine_threadsafe(
                    context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=full_text,
                    ),
                    loop,
                )

        return "".join(answer_parts)

    try:
        # Run the streaming call in a separate thread
        answer = await asyncio.to_thread(_stream_openai)
    except Exception:
        logger.exception("OpenAI API error during streaming")
        # Replace the placeholder with an error message
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text="Error talking to OpenAI API. Please try again later.",
            )
        except Exception:
            # If edit fails for some reason, fall back to a new message
            await msg.reply_text("Error talking to OpenAI API. Please try again later.")
        return

    # Store assistant reply in DB
    add_message_db(conv_id, "assistant", answer)

    # Final formatted edit with MarkdownV2 so ```code``` renders properly
    formatted = format_for_telegram_markdown_v2(answer)

    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=formatted,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception:
        # Fallback: send as a new message if edit fails (e.g. too long)
        await msg.reply_text(formatted, parse_mode=ParseMode.MARKDOWN_V2)


async def post_init(app: Application) -> None:
    """
    Set Telegram command list so clients show them as suggestions
    when user types '/'.
    """
    try:
        await app.bot.set_my_commands(
            [
                BotCommand("start", "Show a short info about this bot"),
                BotCommand("new", "Reset conversation in this chat/topic"),
                BotCommand("topic_prompt", "Set custom persona for this chat/topic"),
                BotCommand("topic_prompt_reset", "Reset persona for this chat/topic"),
                BotCommand("img", "Generate an image from description"),
            ]
        )
        logger.info("Bot commands registered with Telegram")
    except Exception:
        logger.exception("Failed to set bot commands")


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    init_db()

    global ALLOWED_USERS, ALLOWED_GROUPS
    ALLOWED_USERS, ALLOWED_GROUPS = load_whitelists_from_conf()

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_conv))
    app.add_handler(CommandHandler("topic_prompt", topic_prompt))
    app.add_handler(CommandHandler("topic_prompt_reset", topic_prompt_reset))
    app.add_handler(CommandHandler(["img", "image"], generate_image))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chatgpt_handler))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

