import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

tokenize = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
with open('/home/gregorsamsa/Masters Degree/Natural Language Processing/nlp-text-summarizer/NewsAll_Turkey-Syria.txt') as newsfile:
    text = newsfile.read()
