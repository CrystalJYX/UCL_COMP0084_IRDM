# this file runs around 10 min
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


# qid       the query id
# query     the query text
test_queries = pd.read_csv('test-queries.tsv', sep='\t', header=None, names=['qid','query'])
clean_test_queries = task1.preprocessing(test_queries['query'], stopword_removal = True, lemma = True)
qid_query_dict = dict(zip(test_queries.qid,clean_test_queries))


qid_list = list(candidate_passages.qid) # qid
pid_list = list(candidate_passages['pid']) # pid
def qid_to_pid(qid,qid_list):
    # ind  a list contains all pids with the same qid
    index = [i for i,x in enumerate(qid_list) if x == qid ]
    return index


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

def smoothing(qid,inverted_index,method,parameter):
    query = qid_query_dict[qid]
    index = qid_to_pid(qid,qid_list)
    V = len(inverted_index.keys())
    score_list = {}
    for i in index:
        pid = pid_list[i]
        passage = pid_passage_dict[pid]
        D = len(passage) # passage length
        score = 0
        for token in query:
            if token in passage:
                m =  inverted_index[token][pid]
                cqi = sum(inverted_index[token].values())
            else:
                m = 0
                cqi = 0
            
        if (method == 'laplace'):
            score += np.log((m+1)/(D+V)) 
        elif (method == 'lidstone'):
            score += np.log((m+parameter)/(D+V*parameter)) 
        elif (method == 'dirichlet'):
            score += np.log((D/(D+parameter))*(m/D)+(parameter/(parameter+D))*(cqi/V))
        
        Scores = {pid:score}
        score_list = {**score_list ,**Scores}
        top100_score = dict(sorted(score_list.items(), key=lambda x: x[1], reverse=True)[:100])
    return top100_score


with open('laplace.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    queries = qid_query_dict.keys()
    for qid in queries:
        top100_score = smoothing(qid,inverted_index,method = 'laplace',parameter=1)
        for i in range(len(top100_score)):
            writer.writerow([qid,list(top100_score.keys())[i],list(top100_score.values())[i]])

            
with open('lidstone.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    queries = qid_query_dict.keys()
    for qid in queries:
        top100_score = smoothing(qid,inverted_index,method = 'lidstone',parameter=0.1)
        for i in range(len(top100_score)):
            writer.writerow([qid,list(top100_score.keys())[i],list(top100_score.values())[i]])

            
with open('dirichlet.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    queries = qid_query_dict.keys()
    for qid in queries:
        top100_score = smoothing(qid,inverted_index,method = 'dirichlet',parameter=50)
        for i in range(len(top100_score)):
            writer.writerow([qid,list(top100_score.keys())[i],list(top100_score.values())[i]])            
            
            
            
