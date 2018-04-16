import pymorphy2 as pm
import re
from nltk.tokenize import TweetTokenizer
import nltk


morph = pm.MorphAnalyzer()
tokenizer = TweetTokenizer()

# VK account regexp
vk_regexp = r"(https?:\/\/)?(www\.)?(new\.)?(m\.)?vk\.com\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
vklink = "vklink"

gramem_list = ['NOUN', "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"]

stopwords = nltk.corpus.stopwords.words('russian')
stopwords += ["мочь", "пожалуйста", "сбербанк", "быть", "банк", "банка"]

#Lemmatization
def lemmatize_strings(string_list):
    new_list = [morph.parse(string)[0].normal_form for string in string_list if (morph.parse(string)[0].tag.POS in gramem_list or string == vklink) and not in stopwords]
    return new_list

# Preprocessing
def preprocess_question(question):
    question = question.lower() # convert to lower case
    question = re.sub(vk_regexp, vklink, question) # convert vk links to special word
    question = tokenizer.tokenize(question) # tokenize string
    question = lemmatize_strings(question) # lemmatize question
    new_question = ""
    for string in question:
        new_question += string + " "
    new_question = new_question.strip()
    return new_question

if __name__ == "__main__":
    print(preprocess_question("https://vk.com/id393445899 - злой клон почему админа группы = мошенник!!!!!!!!!!"))