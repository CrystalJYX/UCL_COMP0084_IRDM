# this file runs around 8 min
import task1
import pandas as pd
import numpy as np
import csv

passage_collection = task1.read_file('passage-collection.txt')
clean_passage_collection = task1.preprocessing(passage_collection, stopword_removal = True, lemma = True)
vocabulary_list,times = task1.word_occurrence(clean_passage_collection)
len(vocabulary_list) # 144060 vocabulary


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

# test
#print(inverted_index['subclass'])
#print(np.sum(inverted_index['subclass'].values()))
#print(inverted_index['subclass'].keys())

