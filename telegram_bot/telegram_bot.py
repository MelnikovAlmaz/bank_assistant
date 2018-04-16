# coding: utf-8
# more examples: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/README.md
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

TG_TOKEN = "569776209:AAGlS4OT7jFw3oMtQ9781anRKwLtgCKKbNAr"


def idle_main(bot, update):
    bot.sendMessage(update.message.chat_id, text=update.message.text)

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
    main()
