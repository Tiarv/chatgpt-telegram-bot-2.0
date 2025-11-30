# chatgpt-telegram-bot-2.0
A telegram bot that provides access to an OpenAI' LLMs

Designed to work inside a supergroup with topics: to imitate tabs inside chatgpt

NB! You need your organization to be verified by the OpenAI in order for 
streaming to work properly with recent models, also to have access to 
gpt-image-1 model through an API.

## How to install

```bash
useradd chatgpt
cp git/chatgpt-telegram-bot-2.0/etc/systemd/system/chatgpt.service /etc/systemd/system/
su - chatgpt
python3 -m venv venv
. venv/bin/activate
pip install python-telegram-bot openai PyPDF2
cp git/chatgpt-telegram-bot-2.0/bot.py /home/chatgpt/bot.py
cp git/chatgpt-telegram-bot-2.0/bot.conf /home/chatgpt/bot.conf
^D
cat > /etc/chatgpt.env <<EOF
TELEGRAM_BOT_TOKEN=1234567890
OPENAI_API_KEY=0987654321
OPENAI_MODEL="gpt-5.1-2025-11-13"
BOT_DB_PATH=/home/chatgpt/state/conversations.db
OPENAI_IMAGE_MODEL=gpt-image-1
OPENAI_SUMMARY_MODEL="gpt-5.1-2025-11-13"
OPENAI_PROFILE_MODEL="gpt-5.1-2025-11-13"
OPENAI_IMAGE_PROMPT_MODEL="gpt-5.1-2025-11-13"
OPENAI_VISION_MODEL="gpt-5.1-2025-11-13"
EOF
chmod 0400 /etc/chatgpt.env
chown -R chatgpt:chatgpt /home/chatgpt
chmod 0400 /home/chatgpt/bot.py
chmod 0600 /home/chatgpt/bot.conf
systemctl daemon-reload
systemctl enable chatgpt
systemctl start chatgpt
```
