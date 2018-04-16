# Bank Assistant
Bank assistant is project that automatically provide possible answers to client's questions.
It is implemented in Telegram bot format. Username of bot is **@Sberbank_assistant_testbot**
## Installation
Clone project from GitHub repository.

Install requirement packages by pip:

**pip install -r requirements.txt**

### Start project
Use command:

**python telegram_bot.py**

## Project structure
Project consists of 3 parts:
1. Telegram bot
2. Assistant
3. Model training

Telegram bot is simple, just realize one function, accept message and return answer.

Assistant realize 2 function:
1. Find 5 nearest clusters and return information in format {key_words}, {confidence}.
Method name is **_get_nearest_clusters(self, query, num_clusters)_**

2. Find 5 similar questions and return information in format {question}, {answer}, {confidence}.
Method name is **_get_nearest_questions(self, query, num_clusters)_**

### Model training
Model pipeline structure is:
1. Load data
2. Preprocess data and filter
3. Convert documents to vectors by TF-IDF vectorizer
4. Find clusters by KMeans algorithm

#### Preprocess and filter
Preprocessing operations are implemented in _assistant/training/preprocess.py_ module.
There are used lemmatization ,removing stopwords, converting vk links to word "vklink".
