from assistant.training.assistant_train import AssistantTrainer


class Assistant:
    assistant_trainer = AssistantTrainer()

    def get_topic_name(self, query):
        vector = self.assistant_trainer.vectorize_query(query)
        clf = self.assistant_trainer.clf
        topic_id = clf.predict(vector)[0]

        return topic_id


if __name__ == "__main__":
    assistant = Assistant()
    print(assistant.get_topic_name(", у я есть карта  молодёжный . на нея быть отправить средство . как я войти в сбербанконлайн , чтобы проверить , есть ли они на карта , если у я нет ни логин ни пароль ? по телефон позвонить не мочь - нет тоновый набор ."))