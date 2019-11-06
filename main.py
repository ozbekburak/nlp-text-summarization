""" Installing necessary libraries. """
import nltk
import codecs
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

""" 
    Downloading corporas.
    
    We only need it once.
    
    punkt: Tokenizer. Divides a text into a list of sentences. 
    reference: https://www.nltk.org/_modules/nltk/tokenize/punkt.html
    
    stopwords: removing stopwords like `the` `is` `are`
    reference: https://www.nltk.org/nltk_data/
"""

with codecs.open('Summaries/Group1/Group1_News_All.txt', 'r', encoding='utf-8', errors='ignore') as group1_news_all:
    document = group1_news_all.read()

# tokenization
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
tokenized_to_sentence = tokenizer.tokenize(document)

regex_tokenizer = RegexpTokenizer("[\w']+")
tokenized_to_word = regex_tokenizer.tokenize(''.join(tokenized_to_sentence))


def pre_processing(text):
    # deleting stopwords
    english_stops = set(stopwords.words('english'))
    tokenized_words_without_stopwords = []
    for word in tokenized_to_word:
        if word not in english_stops:
            tokenized_words_without_stopwords.append(word)

    # stemming operation
    stemmer = PorterStemmer()
    tokenized_words_without_stopwords_and_stemmed = []
    for word in tokenized_words_without_stopwords:
        tokenized_words_without_stopwords_and_stemmed.append(stemmer.stem(word))

    return tokenized_words_without_stopwords_and_stemmed


def create_word_occurrence_table(tokenized_words_without_stopwords_and_stemmed):
    number_of_occurrences = {}
    for word in tokenized_words_without_stopwords_and_stemmed:
        if word in number_of_occurrences:
            number_of_occurrences[word] += 1
        else:
            number_of_occurrences[word] = 1
    return number_of_occurrences


""" 
    Calculating sentence score.
    
    After creating word_occurrence_table, stopwords does not consider when assign score to sentences. 
"""


def calculate_sentence_score(number_of_occurrences):
    sentence_score = {}
    for sentence in tokenized_to_sentence:
        word_count_in_sentence_without_stopwords = 0
        for word_value in number_of_occurrences:
            if word_value in sentence.lower():  # handling case sensitivity
                word_count_in_sentence_without_stopwords += 1
                if sentence in sentence_score:
                    sentence_score[sentence] += number_of_occurrences[word_value]
                else:
                    sentence_score[sentence] = number_of_occurrences[word_value]
        if sentence in sentence_score:
            sentence_score[sentence] = sentence_score[sentence] / word_count_in_sentence_without_stopwords
    for sentence in sentence_score:
        print(sentence, " : ", round(sentence_score[sentence], 3))

    return sentence_score
