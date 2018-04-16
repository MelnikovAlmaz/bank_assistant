import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from assistant.training.preprocess.preprocess import filter_question, preprocess_question
from assistant.settings import BASE_DIR


class AssistantTrainer:
    base_data_path = "data/base_data_vk.csv"
    prepared_data_path = "data/processed_data_vk.csv"
    prepared_field_name = 'process'

    def preprocess_query(self, query):
        preprocess_string = preprocess_question(query)
        return preprocess_string

    def prepare_data(self):
        data_frame = pd.read_csv(self.base_data_path)
        data_frame = data_frame.dropna()
        data_frame['is_accepted'] = data_frame['question'].apply(lambda x: filter_question(x))
        data_frame = data_frame[data_frame['is_accepted'] == True]
        data_frame.drop(['is_accepted'], axis=1, inplace=True)

        # Preprocess
        data_frame[self.prepared_field_name] = data_frame['question'].apply(lambda x: preprocess_question(x))
        data_frame = data_frame[data_frame[self.prepared_field_name].apply(lambda x: len(x.split()) > 0)]
        data_frame.reset_index(drop=True, inplace=True)

        data_frame.to_csv("processed_df.csv")
        return data_frame

    def load_prepared_data(self):
        data_frame = pd.read_csv(self.prepared_data_path)
        return data_frame

    def prepare_vectorizer(self):
        data_frame = self.load_prepared_data()

        vectorizer = TfidfVectorizer(max_features=10000, max_df=0.5, norm='l2', ngram_range=(1, 2))
        vectorizer.fit(data_frame[self.prepared_field_name])
        joblib.dump(vectorizer, os.path.join(BASE_DIR, "assistant/prepared_modules/tfidf_vectorizer_10000_ngram_12.pkl"))

        return vectorizer

    def load_vectorizer(self):
        vectorizer = joblib.load(os.path.join(BASE_DIR, "assistant/training/tfidf_vectorizer_10000_ngram_12.pkl"))
        return vectorizer

    def vectorize_query(self, query):
        preprocessed_query = preprocess_question(query)
        vector = self.vectorizer.transform([preprocessed_query])
        return vector

    def get_train_vectors(self):
        data_frame = self.load_prepared_data()
        vectorizer = self.load_vectorizer()

        return vectorizer.transform(data_frame[self.prepared_field_name])

    def prepare_clustering(self):
        num_clusters = 60
        clf = KMeans(init='k-means++', max_iter=1000, n_clusters=num_clusters, n_init=15, n_jobs=2, random_state=241)
        vectors = self.get_train_vectors()
        clf.fit(vectors)
        joblib.dump(clf, os.path.join(BASE_DIR, "assistant/prepared_modules/kmeans_60.pkl"))
        return clf

    def load_clustering(self):
        clf = joblib.load(os.path.join(BASE_DIR, "assistant/prepared_modules/kmeans_60.pkl"))
        return clf

    def __init__(self):
        self.vectorizer = self.load_vectorizer()
        self.clf = self.load_clustering()
        print("Ready!")

if __name__ == "__main__":
    assistant = AssistantTrainer()
    vect = assistant.load_vectorizer()