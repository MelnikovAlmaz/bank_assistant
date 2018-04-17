from sklearn.neighbors import NearestNeighbors

from assistant.training.assistant_train import AssistantTrainer


class Assistant:
    assistant_trainer = AssistantTrainer()

    def get_nearest_clusters(self, query, num_clusters):
        """
        Method find similar clusters for user questions.
        :param query: Question from user, type: string
        :param num_clusters: Number of clusters to show, type: int
        :return: List of similar clusters. type: list of dicts, each dict in format
        {'index': , 'name': , 'confidence': }
        """
        # Find cluster for question
        vector = self.assistant_trainer.vectorize_query(query)  # Convert question to vector
        clf = self.assistant_trainer.clf  # Load clustering model
        cluster_id = clf.predict(vector)[0]  # Predict cluster

        # Find cluster centroid vector
        centroid = clf.cluster_centers_.argsort()[cluster_id:cluster_id + 1, ::-1]

        centroids = self.assistant_trainer.clf.cluster_centers_.argsort()  # Get all centroids
        knn = NearestNeighbors(metric='euclidean', algorithm='brute')
        knn.fit(centroids)

        distances, indices = knn.kneighbors(centroid, n_neighbors=num_clusters)  # Find nearest clusters to centroid
        distances = distances.flatten()  # Convert to 1-d list
        nearest_neighbors_index = indices.flatten()  # Convert to 1-d list
        nearest_centroids = centroids[nearest_neighbors_index]

        # Load vocabulary of vectorizer
        vocabulary = self.assistant_trainer.vectorizer.get_feature_names()

        # Find top 3 bigrams for cluster
        key_words = ""
        key_word_count = 0
        for ind in centroid[0]:
            if len(vocabulary[ind].split()) > 1:
                key_words += ' %s' % vocabulary[ind]
                key_word_count += 1
                if key_word_count == 3:
                    break
        cluster_list = [{'index': cluster_id, 'name': key_words, 'confidence': 100}]

        # Process nearest clusters
        for i in range(num_clusters - 1):
            cluster = {'index': nearest_neighbors_index[i], 'name': "", 'confidence': 0}

            centroid = nearest_centroids[i]

            # Find top 3 bigrams for cluster
            key_words = ""
            key_word_count = 0
            for ind in centroid:  # replace 6 with n words per cluster
                if len(vocabulary[ind].split()) > 1:
                    key_words += ' %s' % vocabulary[ind]
                    key_word_count += 1
                    if key_word_count == 3:
                        break
            cluster['name'] = key_words
            cluster['confidence'] = (distances[-1] - distances[i])/(distances[-1] - distances[0]) * 100

            cluster_list.append(cluster)

        return cluster_list

    def get_nearest_questions(self, query, num_clusters):
        """
        Method find similar questions for user question and provide answer with some confidence.

        :param query: Question from user, type: string
        :param num_clusters: Number of clusters to show, type: int
        :return: List of similar questions. type: list of dicts, each dict in format
        {'question': , 'answer': , 'confidence': }
        """
        vector = self.assistant_trainer.vectorize_query(query)  # Convert question to vector
        data_frame = self.assistant_trainer.load_prepared_data()  # Load train data frame

        # Fit KNN model by train data
        knn = NearestNeighbors(metric='euclidean', algorithm='brute')
        knn.fit(self.assistant_trainer.vectorizer.transform(data_frame[self.assistant_trainer.prepared_field_name]))

        distances, indices = knn.kneighbors(vector, n_neighbors=num_clusters + 1)  # Find nearest questions
        indices = indices.flatten()
        distances = distances.flatten()

        question_list = []
        for i in range(num_clusters):
            question_row = data_frame.loc[indices[i]]
            question = [{'question': question_row['question'], 'answer': question_row['answer'], 'confidence': (distances[-1] - distances[i])/(distances[-1] - distances[0]) * 100}]
            question_list.append(question[0])

        return question_list

if __name__ == "__main__":
    assistant = Assistant()
    print(assistant.get_topic_name(", у я есть карта  молодёжный . на нея быть отправить средство . как я войти в сбербанконлайн , чтобы проверить , есть ли они на карта , если у я нет ни логин ни пароль ? по телефон позвонить не мочь - нет тоновый набор ."))