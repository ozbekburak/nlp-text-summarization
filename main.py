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

# kelime sıklıklarını dictionary'de toplayalım.
frequency_table = {}
for word in tokenized_words_without_stopwords_and_stemmed:
    if word in frequency_table:
        frequency_table[word] += 1
    else:
        frequency_table[word] = 1

sentence_score = {}
for sentence in tokenized_to_sentence:
    word_count_in_sentence_without_stopwords = 0
    for wordValue in frequency_table:
        if wordValue in sentence.lower():
            word_count_in_sentence_without_stopwords += 1
            if sentence in sentence_score:
                sentence_score[sentence] += frequency_table[wordValue]
            else:
                sentence_score[sentence] = frequency_table[wordValue]
    if sentence in sentence_score:
        sentence_score[sentence] = sentence_score[sentence] / word_count_in_sentence_without_stopwords
for sentence in sentence_score:
    print(sentence, " : ", round(sentence_score[sentence], 3))

# özet oranlarının hesaplanması
total_length_of_words = 0
for sentence in sentence_score:
    total_length_of_words += len(regex_tokenizer.tokenize(''.join(sentence)))
print("Total length of words in text: ", total_length_of_words)

length_of_twenty_five_percentage_text = round(total_length_of_words/4)
length_of_forty_percentage_text = round((total_length_of_words / 5) * 2)
length_of_sixty_percentage_text = round((total_length_of_words / 5) * 3)
print("Word count of %25 Summary: ", length_of_twenty_five_percentage_text)
print("Word count of %40 Summary: ", length_of_forty_percentage_text)
print("Word count of %60 Summary: ", length_of_sixty_percentage_text)