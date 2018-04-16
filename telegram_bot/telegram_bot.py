# coding: utf-8
# more examples: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/README.md
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from assistant.assistant import Assistant

TG_TOKEN = "569776209:AAGlS4OT7jFw3oMtQ9781anRKwLtgCKKbNA"

assistant = None


def idle_main(bot, update):
    message = update.message.text
    topic_id = assistant.get_topic_name(message)
    print(topic_id)
    bot.sendMessage(update.message.chat_id, text="" + str(topic_id))

def slash_start(bot, update):
    bot.sendMessage(update.message.chat_id, text="Hi!")

def main():
    updater = Updater(TG_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", slash_start), group=0)
    dp.add_handler(MessageHandler(Filters.text, idle_main))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    assistant = Assistant()
    main()