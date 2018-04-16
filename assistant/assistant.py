from sklearn.neighbors import NearestNeighbors

from assistant.training.assistant_train import AssistantTrainer


class Assistant:
    assistant_trainer = AssistantTrainer()

    def get_nearest_clusters(self, query, num_clusters):
        # Find cluster
        vector = self.assistant_trainer.vectorize_query(query)
        clf = self.assistant_trainer.clf
        cluster_id = clf.predict(vector)[0]

        # Find cluster centroid coords
        centroid = clf.cluster_centers_.argsort()[cluster_id:cluster_id + 1, ::-1]

        centroids = self.assistant_trainer.clf.cluster_centers_.argsort()
        knn = NearestNeighbors(metric='euclidean', algorithm='brute')
        knn.fit(centroids)

        distances, indices = knn.kneighbors(centroid, n_neighbors=num_clusters + 1)
        distances = distances.flatten()
        nearest_neighbors = indices.flatten()
        nearest_centroids = centroids[nearest_neighbors]

        vocabulary = self.assistant_trainer.vectorizer.get_feature_names()

        key_words = ""
        key_word_count = 0
        for ind in centroid[0]:  # replace 6 with n words per cluster
            if len(vocabulary[ind].split()) > 1:
                key_words += ' %s' % vocabulary[ind]
                key_word_count += 1
                if key_word_count == 3:
                    break
        cluster_list = [{'index': cluster_id, 'name': key_words, 'confidence': 1}]

        for i in range(num_clusters):
            cluster = {'index': nearest_neighbors[i], 'name': "", 'confidence': 0}

            centroid = nearest_centroids[i]
            key_words = ""
            key_word_count = 0
            for ind in centroid:  # replace 6 with n words per cluster
                if len(vocabulary[ind].split()) > 1:
                    key_words += ' %s' % vocabulary[ind]
                    key_word_count += 1
                    if key_word_count == 3:
                        break
            cluster['name'] = key_words
            cluster['confidence'] = (1 - distances[i]/distances[-1]) * 100

            cluster_list.append(cluster)

        return cluster_list

    def get_nearest_questions(self, query, num_clusters):
        vector = self.assistant_trainer.vectorize_query(query)
        data_frame = self.assistant_trainer.load_prepared_data()

        knn = NearestNeighbors(metric='euclidean', algorithm='brute')
        knn.fit(self.assistant_trainer.vectorizer.transform(data_frame[self.assistant_trainer.prepared_field_name]))

        distances, indices = knn.kneighbors(vector, n_neighbors=num_clusters + 1)
        indices = indices.flatten()
        distances = distances.flatten()
        question_list = []
        for i in range(num_clusters):
            question_row = data_frame.loc[indices[i]]
            question = [{'question': question_row['question'], 'answer': question_row['answer'], 'confidence': (1 - distances[i]/distances[-1]) * 100}]
            question_list.append(question[0])

        return question_list

if __name__ == "__main__":
    assistant = Assistant()
    print(assistant.get_topic_name(", у я есть карта  молодёжный . на нея быть отправить средство . как я войти в сбербанконлайн , чтобы проверить , есть ли они на карта , если у я нет ни логин ни пароль ? по телефон позвонить не мочь - нет тоновый набор ."))