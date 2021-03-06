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
import matplotlib.pyplot as plt
from google_ngram_downloader import readline_google_store
from simhash import Simhash

""" 
    Downloading corporas.
    We only need it once.
    
    punkt: Tokenizer. Divides a text into a list of sentences. 
    reference: https://www.nltk.org/_modules/nltk/tokenize/punkt.html
    
    stopwords: removing stopwords like `the` `is` `are`
    reference: https://www.nltk.org/nltk_data/
"""


def select_news_group(group_no):
    location_base = "Summaries/Group{}/Group{}_News_All.txt".format(group_no, group_no)
    return location_base


print("Welcome to comparison of summaries between human and machine!! ")
print("Group 1 : Turkey-Syria Offensive")
print("Group 2 : Saudi Arabia Oil Attack")
print("Group 3 : Smart Cities")
print("Enter the number of the group you want to see the summary comparison below.")

get_group_number = input("Which group do you want the summary comparison:  ")
similarity_type = input("Which similarity method do you want the use: (please enter `cosin` or `hashsim`) ")
with codecs.open(select_news_group(get_group_number), 'r', encoding='utf-8', errors='ignore') as news_all:
    document = news_all.read()


# tokenization
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
tokenized_to_sentence = tokenizer.tokenize(document)

regex_tokenizer = RegexpTokenizer("[\w']+")
tokenized_to_word = regex_tokenizer.tokenize(''.join(tokenized_to_sentence))


def pre_processing():
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

    return sentence_score


"""
    Calculation of summary lengths.
    
    We need to do this for comparing machine summary and human summary in different size of text.
"""


def calculate_total_word(sentence_score):
    total_length_of_words = 0
    for sentence in sentence_score:
        total_length_of_words += len(regex_tokenizer.tokenize(''.join(sentence)))
    return total_length_of_words


def calculate_summary_length_25(total_length_of_words):
    length_of_twenty_five_percentage_text = round(total_length_of_words / 4)
    return length_of_twenty_five_percentage_text


def calculate_summary_length_40(total_length_of_words):
    length_of_forty_percentage_text = round((total_length_of_words / 5) * 2)
    return length_of_forty_percentage_text


def calculate_summary_length_60(total_length_of_words):
    length_of_sixty_percentage_text = round((total_length_of_words / 3) * 3)
    return length_of_sixty_percentage_text


"""
    Ordering sentences.
    
    After scoring the sentences, sort them in descending order. high score, first priority.
"""


def order_sentences(sentence_score):
    sorted_sentences = sorted(sentence_score, key=sentence_score.get, reverse=True)
    return sorted_sentences


def generate_summaries(sorted_sentences, length_of_twenty_five_percentage_text, length_of_forty_percentage_text,
                       length_of_sixty_percentage_text):
    # generate summary (%25)
    generate_25_summary = ''
    length_25_summary = 0
    for sentence in sorted_sentences:
        while length_25_summary < length_of_twenty_five_percentage_text:
            generate_25_summary += " " + sentence
            length_25_summary += len(regex_tokenizer.tokenize(''.join(sentence)))
            break

    # generate summary (%40)
    generate_40_summary = ''
    length_40_summary = 0
    for sentence in sorted_sentences:
        while length_40_summary < length_of_forty_percentage_text:
            generate_40_summary += " " + sentence
            length_40_summary += len(regex_tokenizer.tokenize(''.join(sentence)))
            break

    # generate summary (%60)
    generate_60_summary = ''
    length_60_summary = 0
    for sentence in sorted_sentences:
        while length_60_summary < length_of_sixty_percentage_text:
            generate_60_summary += " " + sentence
            length_60_summary += len(regex_tokenizer.tokenize(''.join(sentence)))
            break
    return generate_25_summary, generate_40_summary, generate_60_summary


def read_human_summaries_25():
    with codecs.open('Summaries/Group{}/Group{}_Summaries_25/Group{}_Summary_25_1.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_25_1:
        group_summary_25_1_text = group_summary_25_1.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_25/Group{}_Summary_25_2.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_25_2:
        group_summary_25_2_text = group_summary_25_2.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_25/Group{}_Summary_25_3.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_25_3:
        group_summary_25_3_text = group_summary_25_3.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_25/Group{}_Summary_25_4.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_25_4:
        group_summary_25_4_text = group_summary_25_4.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_25/Group{}_Summary_25_5.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_25_5:
        group_summary_25_5_text = group_summary_25_5.read()

    return group_summary_25_1_text, group_summary_25_2_text, group_summary_25_3_text, group_summary_25_4_text, group_summary_25_5_text


def read_human_summaries_40():
    with codecs.open('Summaries/Group{}/Group{}_Summaries_40/Group{}_Summary_40_1.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_40_1:
        group_summary_40_1_text = group_summary_40_1.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_40/Group{}_Summary_40_2.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_40_2:
        group_summary_40_2_text = group_summary_40_2.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_40/Group{}_Summary_40_3.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_40_3:
        group_summary_40_3_text = group_summary_40_3.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_40/Group{}_Summary_40_4.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_40_4:
        group_summary_40_4_text = group_summary_40_4.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_40/Group{}_Summary_40_5.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_40_5:
        group_summary_40_5_text = group_summary_40_5.read()

    return group_summary_40_1_text, group_summary_40_2_text, group_summary_40_3_text, group_summary_40_4_text, group_summary_40_5_text


def read_human_summaries_60():
    with codecs.open('Summaries/Group{}/Group{}_Summaries_60/Group{}_Summary_60_1.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_60_1:
        group_summary_60_1_text = group_summary_60_1.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_60/Group{}_Summary_60_2.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_60_2:
        group_summary_60_2_text = group_summary_60_2.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_60/Group{}_Summary_60_3.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_60_3:
        group_summary_60_3_text = group_summary_60_3.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_60/Group{}_Summary_60_4.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_60_4:
        group_summary_60_4_text = group_summary_60_4.read()
    with codecs.open('Summaries/Group{}/Group{}_Summaries_60/Group{}_Summary_60_5.txt'.format(get_group_number, get_group_number, get_group_number), 'r', encoding='utf-8',
                     errors='ignore') as group_summary_60_5:
        group_summary_60_5_text = group_summary_60_5.read()

    return group_summary_60_1_text, group_summary_60_2_text, group_summary_60_3_text, group_summary_60_4_text, group_summary_60_5_text


"""
    Comparison of machine summary and human summary.
    
    To do that, we use cosine similarity.
    
    References: 
    https://www.machinelearningplus.com/nlp/cosine-similarity/ http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html    
"""


def calculate_similarity_between_documents(machine_summary, human_summary):
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf = vectorizer.fit_transform([machine_summary, human_summary])
    return cosine_similarity(tfidf)


pre_processed_text = pre_processing()

word_occurrence_table = create_word_occurrence_table(pre_processed_text)

calculated_sentence_score = calculate_sentence_score(word_occurrence_table)

calculated_total_word = calculate_total_word(calculated_sentence_score)

calculated_summary_length_25 = calculate_summary_length_25(calculated_total_word)

calculated_summary_length_40 = calculate_summary_length_40(calculated_total_word)

calculated_summary_length_60 = calculate_summary_length_60(calculated_total_word)

ordered_sentences = order_sentences(calculated_sentence_score)


"""
    generated_summaries consist of 3 summaries (%25) (%40) (%60). 
    
    generated_summaries[0] : %25 Summary
    generated_summaries[1] : %40 Summary
    generated_summaries[2] : %60 Summary
"""


generated_summaries = generate_summaries(ordered_sentences, calculated_summary_length_25, calculated_summary_length_40, calculated_summary_length_60)

"""
    group1_summaries_X consist of 5 summaries.
    
    group1_summaries_X[0] : Summary of 1. human.
    group1_summaries_X[1] : Summary of 2. human.
    group1_summaries_X[2] : Summary of 3. human.
    group1_summaries_X[3] : Summary of 4. human.
    group1_summaries_X[4] : Summary of 5. human.    
"""


group_summaries_25 = read_human_summaries_25()
group_summaries_40 = read_human_summaries_40()
group_summaries_60 = read_human_summaries_60()


similarity_ratio_first_person_25 = round(calculate_similarity_between_documents(generated_summaries[0], group_summaries_25[0]).item(1), 3)
similarity_ratio_second_person_25 = round(calculate_similarity_between_documents(generated_summaries[0], group_summaries_25[1]).item(1), 3)
similarity_ratio_third_person_25 = round(calculate_similarity_between_documents(generated_summaries[0], group_summaries_25[2]).item(1), 3)
similarity_ratio_forth_person_25 = round(calculate_similarity_between_documents(generated_summaries[0], group_summaries_25[3]).item(1), 3)
similarity_ratio_fifth_person_25 = round(calculate_similarity_between_documents(generated_summaries[0], group_summaries_25[4]).item(1), 3)


similarity_ratio_first_person_40 = round(calculate_similarity_between_documents(generated_summaries[1], group_summaries_40[0]).item(1), 3)
similarity_ratio_second_person_40 = round(calculate_similarity_between_documents(generated_summaries[1], group_summaries_40[1]).item(1), 3)
similarity_ratio_third_person_40 = round(calculate_similarity_between_documents(generated_summaries[1], group_summaries_40[2]).item(1), 3)
similarity_ratio_forth_person_40 = round(calculate_similarity_between_documents(generated_summaries[1], group_summaries_40[3]).item(1), 3)
similarity_ratio_fifth_person_40 = round(calculate_similarity_between_documents(generated_summaries[1], group_summaries_40[4]).item(1), 3)

similarity_ratio_first_person_60 = round(calculate_similarity_between_documents(generated_summaries[2], group_summaries_60[0]).item(1), 3)
similarity_ratio_second_person_60 = round(calculate_similarity_between_documents(generated_summaries[2], group_summaries_60[1]).item(1), 3)
similarity_ratio_third_person_60 = round(calculate_similarity_between_documents(generated_summaries[2], group_summaries_60[2]).item(1), 3)
similarity_ratio_forth_person_60 = round(calculate_similarity_between_documents(generated_summaries[2], group_summaries_60[3]).item(1), 3)
similarity_ratio_fifth_person_60 = round(calculate_similarity_between_documents(generated_summaries[2], group_summaries_60[4]).item(1), 3)

'''
    Similarity check with hashsim.
    
    Actually, it calculates the distance between the text/documents.
    
'''


similarity_ratio_first_person_25_hashsim = (1 - (Simhash(generated_summaries[0]).distance(Simhash(group_summaries_25[0])) / 64)) * 100
similarity_ratio_second_person_25_hashsim = (1 - (Simhash(generated_summaries[0]).distance(Simhash(group_summaries_25[1])) / 64)) * 100
similarity_ratio_third_person_25_hashsim = (1 - (Simhash(generated_summaries[0]).distance(Simhash(group_summaries_25[2])) / 64)) * 100
similarity_ratio_forth_person_25_hashsim = (1 - (Simhash(generated_summaries[0]).distance(Simhash(group_summaries_25[3])) / 64)) * 100
similarity_ratio_fifth_person_25_hashsim = (1 - (Simhash(generated_summaries[0]).distance(Simhash(group_summaries_25[4])) / 64)) * 100


similarity_ratio_first_person_40_hashsim = (1 - (Simhash(generated_summaries[1]).distance(Simhash(group_summaries_40[0])) / 64)) * 100
similarity_ratio_second_person_40_hashsim = (1 - (Simhash(generated_summaries[1]).distance(Simhash(group_summaries_40[1])) / 64)) * 100
similarity_ratio_third_person_40_hashsim = (1 - (Simhash(generated_summaries[1]).distance(Simhash(group_summaries_40[2])) / 64)) * 100
similarity_ratio_forth_person_40_hashsim = (1 - (Simhash(generated_summaries[1]).distance(Simhash(group_summaries_40[3])) / 64)) * 100
similarity_ratio_fifth_person_40_hashsim = (1 - (Simhash(generated_summaries[1]).distance(Simhash(group_summaries_40[4])) / 64)) * 100


similarity_ratio_first_person_60_hashsim = (1 - (Simhash(generated_summaries[2]).distance(Simhash(group_summaries_60[0])) / 64)) * 100
similarity_ratio_second_person_60_hashsim = (1 - (Simhash(generated_summaries[2]).distance(Simhash(group_summaries_60[1])) / 64)) * 100
similarity_ratio_third_person_60_hashsim = (1 - (Simhash(generated_summaries[2]).distance(Simhash(group_summaries_60[2])) / 64)) * 100
similarity_ratio_forth_person_60_hashsim = (1 - (Simhash(generated_summaries[2]).distance(Simhash(group_summaries_60[3])) / 64)) * 100
similarity_ratio_fifth_person_60_hashsim = (1 - (Simhash(generated_summaries[2]).distance(Simhash(group_summaries_60[4])) / 64)) * 100


data_frame_summarization = pd.DataFrame.from_dict({
    'Machine (%25)': [round(similarity_ratio_first_person_25, 3),
                      round(similarity_ratio_second_person_25, 3),
                      round(similarity_ratio_third_person_25, 3),
                      round(similarity_ratio_forth_person_25, 3),
                      round(similarity_ratio_fifth_person_25, 3)],
    'Machine (%40)': [round(similarity_ratio_first_person_40, 3),
                      round(similarity_ratio_second_person_40, 3),
                      round(similarity_ratio_third_person_40, 3),
                      round(similarity_ratio_forth_person_40, 3),
                      round(similarity_ratio_fifth_person_40, 3)],
    'Machine (%60)': [round(similarity_ratio_first_person_60, 3),
                      round(similarity_ratio_second_person_60, 3),
                      round(similarity_ratio_third_person_60, 3),
                      round(similarity_ratio_forth_person_60, 3),
                      round(similarity_ratio_fifth_person_60, 3)]
}, orient='index', columns=['Human-1', 'Human-2', 'Human-3', 'Human-4', 'Human-5'])


data_frame_summarization_hashsim = pd.DataFrame.from_dict({
    'Machine (%25)': [similarity_ratio_first_person_25_hashsim,
                      similarity_ratio_second_person_25_hashsim,
                      similarity_ratio_third_person_25_hashsim,
                      similarity_ratio_forth_person_25_hashsim,
                      similarity_ratio_fifth_person_25_hashsim],
    'Machine (%40)': [similarity_ratio_first_person_40_hashsim,
                      similarity_ratio_second_person_40_hashsim,
                      similarity_ratio_third_person_40_hashsim,
                      similarity_ratio_forth_person_40_hashsim,
                      similarity_ratio_fifth_person_40_hashsim],
    'Machine (%60)': [similarity_ratio_first_person_60_hashsim,
                      similarity_ratio_second_person_60_hashsim,
                      similarity_ratio_third_person_60_hashsim,
                      similarity_ratio_forth_person_60_hashsim,
                      similarity_ratio_fifth_person_60_hashsim]
}, orient='index', columns=['Human-1', 'Human-2', 'Human-3', 'Human-4', 'Human-5'])


N = 5
if similarity_type == "cosin":
    summaries_25 = (data_frame_summarization.values[0][0]*100, data_frame_summarization.values[0][1]*100,
                    data_frame_summarization.values[0][2]*100, data_frame_summarization.values[0][3]*100,
                    data_frame_summarization.values[0][4]*100)
elif similarity_type == "hashsim":
    summaries_25 = (data_frame_summarization_hashsim.values[0][0], data_frame_summarization_hashsim.values[0][1],
                    data_frame_summarization_hashsim.values[0][2], data_frame_summarization_hashsim.values[0][3],
                    data_frame_summarization_hashsim.values[0][4])
else:
    print("Invalid similarity type entered!! Try again.")


fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.25          # the width of the bars
p1 = ax.bar(ind, summaries_25, width, bottom=0)

if similarity_type == "cosin":
    summaries_40 = (data_frame_summarization.values[1][0]*100, data_frame_summarization.values[1][1]*100,
                    data_frame_summarization.values[1][2]*100, data_frame_summarization.values[1][3]*100,
                    data_frame_summarization.values[1][4]*100)
elif similarity_type == "hashsim":
    summaries_40 = (data_frame_summarization_hashsim.values[1][0], data_frame_summarization_hashsim.values[1][1],
                    data_frame_summarization_hashsim.values[1][2], data_frame_summarization_hashsim.values[1][3],
                    data_frame_summarization_hashsim.values[1][4])
else:
    print("Invalid similarity type entered!! Try again.")
p2 = ax.bar(ind + width, summaries_40, width, bottom=0)

if similarity_type == "cosin":
    summaries_60 = (data_frame_summarization.values[2][0]*100, data_frame_summarization.values[2][1]*100,
                    data_frame_summarization.values[2][2]*100, data_frame_summarization.values[2][3]*100,
                    data_frame_summarization.values[2][4]*100)
elif similarity_type == "hashsim":
    summaries_60 = (data_frame_summarization_hashsim.values[2][0], data_frame_summarization_hashsim.values[2][1],
                    data_frame_summarization_hashsim.values[2][2], data_frame_summarization_hashsim.values[2][3],
                    data_frame_summarization_hashsim.values[2][4])
else:
    print("Invalid similarity type entered!! Try again.")
p3 = ax.bar(ind + width*2, summaries_60, width, bottom=0)

ax.set_title('Similarity comparison ({}) percentage for Group{} (%)'.format(similarity_type ,get_group_number))
ax.set_xticks(ind + width)
ax.set_xticklabels(('Human1', 'Human2', 'Human3', 'Human4', 'Human5'))

ax.legend((p1[0], p2[0], p3[0]), ('%25', '%40', '%60'))
ax.autoscale_view()

plt.show()
