# NLP Text-Summarization

**The Goal**

Compare human summaries and computer summaries from text files compiled from different
news headlines.

**Headlines**

1. Turkey-Syria Offensive
2. Saudi Arabia Oil Attack
3. Smart Cities

**Method**

* Collecting news from different sources
* Creation of human summaries, according to the importance of news
    * For each news 5 people created summaries. This means, for 3 topics 15 different people created
    summary.
* Based on the total number of words,
    * %25 of the summaries were created
    * %40 of the summaries were created
    * %60 of the summaries were created
* After the human summaries are completed, we gave all the summary to the computer.
* First, we did some data preprocessing
    * Tokenizing
    * Removing stopwords
    * Stemming
* After data preprocessing we did some calculations
    * Counting total words
    * Creating word occurrence table (how many words the sentences contain)
    * We assigned score to the sentences (using word occurrence table and total number of words in a sentence)
* We have created sub-summaries (%25, %40, %60)
* Lastly, we compared our results between our human summary and computer summary.

**References**

Balabantaray, R.C., Sahoo, D.K., Sahoo, B., Swain, M. (2012). **Text Summarization Using Term Weights**

Zaefarian, Reza (2006). **A New Algorithm for Term Weighting in Text Summarization Process**

Kumar, Yogan Jaya. Goh, Ong Sing, Basiron, Halizah, Choon, Ngo Hea, Suppiah, Puspalata C(2016). **A review on automatic text summarization approaches**

Hingu, Dharmendra, Shah, Deep, Udmale, Sandeep S(2015).**Automatic text summarization of Wikipedia articles**

Ramaswamy, Sridhar, DeClerck, Natalie(2018). **Customer perception analysis using deep learning and NLP**