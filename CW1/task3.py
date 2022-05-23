# this file runs around 40 min
import task1
import pandas as pd
import numpy as np
import csv
import collections
from collections import Counter

# qid       the query id
# pid       the passage id
# query     the query text
# passage   the passage text
candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names=['qid','pid','query','passage'])
clean_candidate_passages = task1.preprocessing(candidate_passages.passage, stopword_removal = True, lemma = True)
pid_passage_dict = dict(zip(candidate_passages.pid,clean_candidate_passages))


def inv_index(pid_passage_dict):
    inverted_index = {}
    pid_dict = pid_passage_dict.keys()
    for pid in pid_dict:
        passage = pid_passage_dict[pid]
        for token in passage:
            num = passage.count(token)
            if token not in inverted_index.keys():
                inverted_index[token] = {pid:num} 
            elif token in inverted_index.keys():
                new_pid = {pid: num}
                inverted_index[token].update(new_pid) 
    return inverted_index

inverted_index = inv_index(pid_passage_dict)


def tf_idf_passage(pid_passage_dict,inverted_index):
    pid_dict = pid_passage_dict.keys()
    N = len(pid_dict) # number of documents in collection
    TF_IDF_dict = {}
    IDF_dict = {}
    for pid in pid_dict:
        passage = pid_passage_dict[pid]
        passage_tf_idf = {}
        passage_idf = {}
        for token in passage:
            # number of token appears in current passage / total number of words in this passage
            tf = inverted_index[token][pid]/len(passage) 
            n = len(inverted_index[token]) # number of documents in which token appears
            idf = np.log10(N/n)
            #print(idf)
            if token not in passage_idf.keys():
                passage_idf[token] = idf                 
            tf_idf = tf*idf            
            if token not in passage_tf_idf.keys():
                passage_tf_idf[token] = tf_idf
        TF_IDF = {pid:passage_tf_idf}
        IDF = {pid:passage_idf}
        #print(IDF)
        #print(TF_IDF)
        TF_IDF_dict = {**TF_IDF_dict,**TF_IDF} 
        IDF_dict = {**IDF_dict,**IDF}  
        
    return TF_IDF_dict,IDF_dict


# this fuction runs about 20-30 min
#import time
#time_start = time.time()
TF_IDF_dict, IDF_dict = tf_idf_passage(pid_passage_dict,inverted_index)
#time_end = time.time()
#time_c= time_end - time_start
#print('time cost', time_c, 's')

# test
#IDF_dict[8001869]
#TF_IDF_dict[8001869]
#TF_IDF_dict[8001869]['strateg']


# qid       the query id
# query     the query text
test_queries = pd.read_csv('test-queries.tsv', sep='\t', header=None, names=['qid','query'])
clean_test_queries = task1.preprocessing(test_queries['query'], stopword_removal = True, lemma = True)
qid_query_dict = dict(zip(test_queries.qid,clean_test_queries))


def tf_query(qid_query_dict,inverted_index_query):
    qid_dict = qid_query_dict.keys()
    TF_dict = {}
    for qid in qid_dict: 
        query = qid_query_dict[qid]
        query_tf = {}
        for token in query:
            tf = inverted_index_query[token][qid]/len(query) 
            if token not in query_tf.keys():
                query_tf[token] = tf                            
        TF = {qid:query_tf}
        TF_dict = {**TF_dict,**TF}  
    return TF_dict

inverted_index_query = inv_index(qid_query_dict)
query_TF_dict = tf_query(qid_query_dict,inverted_index_query)


qid_list = list(candidate_passages.qid)
pid_list = list(candidate_passages['pid'])
def qid_to_pid(qid,qid_list):
    # index  a list contains all pids with the same qid
    index = [i for i,x in enumerate(qid_list) if x == qid ]
    return index


def tf_idf_query(query_TF_dict, IDF_dict, qid):
    ind = qid_to_pid(qid,qid_list)
    tf = query_TF_dict[qid]
    k1 = set(list(tf.keys()))
    TF_IDF_dict = {}
    for i in ind:
        pid = pid_list[i]
        idf = IDF_dict[pid]
        k2 = set(list(idf.keys()))
        k_c = k1 & k2
        query_tfidf = {}
        for token in k_c:
            tf_itf = tf[token]*idf[token]
            if token not in query_tfidf.keys():
                query_tfidf[token] = tf_itf  
        TF_IDF = {pid:query_tfidf}
        TF_IDF_dict = {**TF_IDF_dict ,**TF_IDF}
    return TF_IDF_dict

TFIDF_passage_dict = TF_IDF_dict


def cosine_similarity(v1, v2):
    # v1  dict of queries
    # v2  dict of passages
    k1 = set(list(v1.keys()))
    k2 = set(list(v2.keys()))
    k_c = k1 & k2 # same words
    inner_product = 0
    for word in k_c:
        inner_product += v1[word] * v2[word]
    L1 = np.linalg.norm(list(v1.values())) 
    L2 = np.linalg.norm(list(v2.values()))
    if inner_product == 0:
        sol = 0
    else:
        sol = inner_product / (L1 * L2)  
    return sol   


# for each qid
def cosine_tfidf_top100(qid):
    query_vector = tf_idf_query(query_TF_dict, IDF_dict, qid)
    top100_qid = {}
    for pid in query_vector.keys():
        v1 = TFIDF_passage_dict[pid]
        v2 = query_vector[pid]
        score = cosine_similarity(v1, v2)
        score_list = {pid:score}
        top100_qid = {**top100_qid ,**score_list}
        top100_score = dict(sorted(top100_qid.items(), key=lambda x: x[1], reverse=True)[:100])
    return top100_score


with open('tfidf.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    queries = qid_query_dict.keys()
    for qid in queries:
        top100_score = cosine_tfidf_top100(qid)
        for i in range(len(top100_score)):
            writer.writerow([qid,list(top100_score.keys())[i],list(top100_score.values())[i]])


length = 0
for pid in pid_passage_dict.keys():
    length += len(pid_passage_dict[pid])
avdl = length/len(pid_passage_dict.keys()) # 32.2764
N = len(pid_list) # 189877


def BM25(qid,pid_passage_dict,qid_query_dict, inverted_index,k1=1.2,k2=100,b=0.75):
    query = qid_query_dict[qid] # qid query
    pids = qid_to_pid(qid, qid_list)
    count_words_q = Counter(query)
    tokens = count_words_q.most_common(len(count_words_q))
    score_list = {}
    for ind in pids:
        pid = pid_list[ind]
        passage = pid_passage_dict[pid] # passage
        dl = len(passage)
        K = k1*((1-b)+b*dl/avdl)
        score = 0
        for i in range(len(tokens)):
            token = tokens[i][0]
            qfi = tokens[i][1]
            if token in passage:
                fi = inverted_index[token][pid]
                ni = len(inverted_index[token])
            else:
                fi = 0
                ni = 0
            score += np.log( ((0+0.5)/(0-0+0.5)) /((ni-0+0.5)/(N-ni-0))) * (k1+1)*fi*(k2+1)*qfi / ((K+fi)*(k2+qfi))
        Scores = {pid:score}
        score_list = {**score_list ,**Scores}
    top100_score = dict(sorted(score_list.items(), key=lambda x: x[1], reverse=True)[:100])
    return top100_score


with open('bm25.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    queries = qid_query_dict.keys()
    for qid in queries:
        top100_score = BM25(qid,pid_passage_dict,qid_query_dict, inverted_index,k1=1.2,k2=100,b=0.75)
        for i in range(len(top100_score)):
            writer.writerow([qid,list(top100_score.keys())[i],list(top100_score.values())[i]])
