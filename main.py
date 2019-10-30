import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#nltk.download('punkt')
#nltk.download('stopwords')

with open('/home/gregorsamsa/Masters Degree/Natural Language Processing/nlp-text-summarizer/NewsAll_Turkey-Syria.txt') as newsfile:
    text = newsfile.read()

# Metini cumlelee boluyoruz.
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
tokenized_to_sentence = tokenizer.tokenize(text)

# Kelime kelime bolmek icin on hazirlik
regex_tokenizer = RegexpTokenizer("[\w']+")
tokenized_to_word = regex_tokenizer.tokenize(''.join(tokenized_to_sentence))

# stopwordsleri ayristiralim
english_stops = set(stopwords.words('english'))
tokenized_words_without_stopwords = []
for word in tokenized_to_word:
    if word not in english_stops:
        tokenized_words_without_stopwords.append(word)

# stemming isleminin uygulanmasi (kelimelerin kokleri alinmasi
stemmer = PorterStemmer()
tokenized_words_without_stopwords_and_stemmed = []
for word in tokenized_words_without_stopwords:
    tokenized_words_without_stopwords_and_stemmed.append(stemmer.stem(word))
