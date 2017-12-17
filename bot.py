import logging
import requests
from deepdream import DeepDreamer, download_model
from gc import collect
from io import BytesIO
from os import environ
from secrets import token_urlsafe
from telegram import ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram.ext.dispatcher import run_async

# Constants
DREAMER = DeepDreamer()
HOST = environ.get("HOST", "0.0.0.0")
PORT = int(environ.get("PORT", "5000"))
TOKEN = environ.get("TELEGRAM_TOKEN")
URL = environ.get("URL")
URL_PATH = token_urlsafe(20)

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def start(bot, update):
    update.message.reply_text(
        "Hola, este bot aplica el algoritmo conocido como Deep Dream sobre "
        "las imágenes que recibe. Solo hay que enviarle una imagen para "
        "aplicar el algoritmo sobre esta."
    )


@run_async
def dream(bot, update):
    chat_id = update.message.chat_id
    # Get image
    bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    photos = update.message.photo
    image_url = bot.get_file(photos[min(2, len(photos))].file_id).file_path
    r = requests.get(image_url, stream=True)
    r.raw.decode_content = True
    # Dream
    output = DeepDreamer().dream(r.raw)
    # Send image
    output_bytes = BytesIO()
    output.save(output_bytes, "JPEG")
    output_bytes.seek(0)
    bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
    update.message.reply_photo(photo=output_bytes)
    del output
    del output_bytes
    collect()


def error(bot, update, error):
    logger.warning("Update '{}' caused error '{}'", update, error)

if __name__ == '__main__':
    # Prepare bot
    updater = Updater(TOKEN, workers=10)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo & Filters.private, dream))
    dp.add_error_handler(error)
    # Configure bot
    updater.start_webhook(listen=HOST, port=PORT, url_path=URL_PATH)
    updater.bot.setWebhook("{}/{}".format(URL, URL_PATH))
    updater.idle()
