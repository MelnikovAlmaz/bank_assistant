# coding: utf-8
# more examples: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/README.md
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from assistant.assistant import Assistant

TG_TOKEN = "569776209:AAGlS4OT7jFw3oMtQ9781anRKwLtgCKKbNA"

assistant = None


def idle_main(bot, update):
    message = update.message.text
    nearest_clusters = assistant.get_nearest_clusters(message, 5)

    for cluster in nearest_clusters:
        bot.sendMessage(update.message.chat_id, text="Cluster {}: {}, confidence: {}\n\n".format(
            cluster['index'], cluster['name'], cluster['confidence']))

    nearest_questions = assistant.get_nearest_questions(message, 5)
    for question in nearest_questions:
        bot.sendMessage(update.message.chat_id, text="Question {},\nAnswer: {},\nConfidence: {}\n\n".format(
            question['question'], question['answer'], question['confidence']))


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