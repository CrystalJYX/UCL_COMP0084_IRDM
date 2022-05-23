# this file runs around 5 min
import string
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer 
import collections
from collections import Counter
from sklearn.linear_model import LinearRegression


def read_file(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        return [line.strip().lower() for line in lines]
       # convert to lowercase letters
       # remove the blanks at the start/end of each line


passage_collection = read_file('passage-collection.txt')
#print(len(passage_collection)) # 182469 passages


# total words
nwords = 0
for i in range(len(passage_collection)):
    words = passage_collection[i]
    # remove all the punctuation
    rm_punc =re.compile('[%s]' % re.escape(string.punctuation))
    words = rm_punc.sub('', words)
    # remove all the numbers
    words = re.sub(r'[^a-zA-Z\s]', u' ', words, flags=re.UNICODE)
    # split into words
    #words = re.split(r'\s+', words)
    word_tokens = RegexpTokenizer(r'\s+', gaps=True)
    words = word_tokens.tokenize(words)
    nwords += len(words)
#print(nwords) # total words 10061726


def preprocessing(text, stopword_removal = False, lemma = False):
    stop_words = set(stopwords.words('english')) 
    word_tokens = RegexpTokenizer(r'\s+', gaps=True)
    passage = []
    for i in range(len(text)):
        words = text[i].lower()
        # remove punctuation
        rm_punc =re.compile('[%s]' % re.escape(string.punctuation))
        words = rm_punc.sub('', words)
        # remove all the numbers
        words = re.sub(r'[^a-zA-Z\s]', u' ', words, flags=re.UNICODE)
        # tokenize
        token_words = word_tokens.tokenize(words)
        
        # stop word removal
        if (stopword_removal == True):
            token_words = [w for w in token_words if not w in stop_words]
        
        sentence = []
        # lemmatisation & stemming
        if (lemma == True):
            stemmer = SnowballStemmer('english')
            for i in token_words:      
                sentence.append(stemmer.stem(i))
        else:
            sentence = token_words
        passage.append(sentence) 
    return passage

passage = preprocessing(passage_collection, stopword_removal = False, lemma = False)


def word_occurrence(passage):
    passages = []
    for i in range(len(passage)):
        passages += passage[i]
    count_words = Counter(passages) # return word and its repeated times
    sorted_words = count_words.most_common(len(count_words)) # sort the list by decreasing times
    
    words = []
    times = []
    for i in sorted_words:
        #word_list = str(i[0]) + ' ' + str(i[1])
        #print(word_list)
        words.append(i[0])
        times.append(i[1])
    return words, times


words, times = word_occurrence(passage)
word_list_len = len(words) # 174911 words in vocabulary list
total = np.sum(times) #  total words 10061726 in this file


plt.figure(figsize = (8,4)) 
rank = range(1000)
freq = times[:1000]/total
plt.plot(rank, freq, color='k')
plt.xlabel('Term Frequency Ranking', fontsize=17)
plt.ylabel('Term Prob. of Occurrence', fontsize=17)
plt.savefig('task1(1).pdf', bbox_inches = 'tight')
plt.show()


plt.figure(figsize = (8,4)) 
log_rank =np.log10(np.arange(1, 1001)).reshape(-1,1)
log_freq =np.log10(freq[:1000]).reshape(-1,1)
model = LinearRegression().fit(log_rank, log_freq)
#print('slope = '+ str(model.coef_)) # slope = [[-0.86428263]]
#print('intercept = ' + str(model.intercept_)) # intercept = [-1.28892898]
pred = model.predict(log_rank)
plt.plot(log_rank, pred, color='k', label='fitted line')
plt.scatter(log_rank,log_freq, color='salmon', label='real data')
plt.legend()
plt.xlabel('Term Frequency Ranking (log10)', fontsize=17)
plt.ylabel('Term Prob. of Occurrence (log10)', fontsize=17)
text = 'log(fitted_freq) = -1.29 - 0.86 * log(freq_rank)'
plt.text(1.1, -2, text, size=11.5)
plt.savefig('task1(2).pdf', bbox_inches = 'tight')
plt.show()
