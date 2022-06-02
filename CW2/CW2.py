# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torchvision
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
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import random
import xgboost as xgb

################################################
## Data prepocessing
################################################

# format : qid pid queries passage relevancy
# 1103039 rows × 5 columns
validation_data = pd.read_csv('validation_data.tsv', sep='\t',header=0,low_memory=False)

def preprocessing(text, stopword_removal = False, lemma = False):
    """
        A text preprocessing function
        Inputs:
          text: input queries/passages
          stopword_removal: remove all stopwords if True
          lemma: do lemmatisation and stemming if True
        Outputs:
          passage: queries/passages after preprocessing
    """
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

# 1103039 rows
clean_validation_query = preprocessing(validation_data.queries, stopword_removal = True, lemma = True)
clean_validation_passage = preprocessing(validation_data.passage, stopword_removal = True, lemma = True)

# 1148 qids and queries
qid_query_valid = dict(zip(validation_data.qid,clean_validation_query))

# 955211 pids and passage
pid_passage_valid = dict(zip(validation_data.pid,clean_validation_passage))

# save files
#np.save('qid_query_valid.npy',qid_query_valid)
#np.save('pid_passage_valid.npy',pid_passage_valid)
#qid_query_valid = np.load('qid_query_valid.npy',allow_pickle=True).tolist()
#pid_passage_valid = np.load('pid_passage_valid.npy',allow_pickle=True).tolist()


################################################
## Task 1 Evaluating Retrieval Quality
################################################
def inv_index(pid_passage_dict):
    """
        A inverted index function
        Inputs:
          pid_passage_dict: dictionary with a format of {pid:passage}
        Outputs:
          inverted_index: dictionary with a format of {token: repeat times}
    """
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

inverted_index = inv_index(pid_passage_valid)
#np.save('inverted_index.npy',inverted_index)
#inverted_index = np.load('inverted_index.npy',allow_pickle=True).tolist()
#inverted_index['start']

length = 0
for pid in pid_passage_valid.keys():
    length += len(pid_passage_valid[pid])
avdl = length/len(pid_passage_valid.keys()) # 32.209788203862814
N = len(validation_data) # 1103039

# 1103039 rows
qid_list = list(validation_data.qid)
pid_list = list(validation_data.pid)
def qid_to_pid(qid,qid_list,pid_list):
    pid = []
    index = [i for i,x in enumerate(qid_list) if x == qid ]
    for i in range(len(index)):
        pid.append(pid_list[index[i]])
    return pid,index

def BM25(qid,pid_passage_dict,qid_query_dict,inverted_index,k1=1.2,k2=100,b=0.75,cutoff=100):
    """
        A BM25 function
        Inputs:
          qid: query id
          pid_passage_dict: dictionary with a format of {pid:passage}
          qid_query_dict: dictionary with a format of {qid:query}
          inverted_index: dictionary with a format of {token: repeat times}
        Outputs:
          top100_score: dictionary with a format of {pid: BM25 scores}
    """
    query = qid_query_dict[qid] # qid query
    pids,ind = qid_to_pid(qid,qid_list,pid_list) # coresponding pid
    count_words_q = Counter(query)
    tokens = count_words_q.most_common(len(count_words_q))
    score_list = {}
    for j in range(len(pids)):
        pid = pids[j]
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
    top100_score = dict(sorted(score_list.items(), key=lambda x: x[1], reverse=True)[:cutoff])
    return top100_score

BM25_score_100 = {}
for qid in qid_query_valid.keys():
    top100_BM25 = BM25(qid,pid_passage_valid,qid_query_valid,inverted_index,k1=1.2,k2=100,b=0.75,cutoff=100)
    Scores = {qid:top100_BM25}
    BM25_score_100 = {**BM25_score_100 ,**Scores}

BM25_score_10 = {}
for qid in qid_query_valid.keys():
    top10_BM25 = BM25(qid,pid_passage_valid,qid_query_valid,inverted_index,k1=1.2,k2=100,b=0.75,cutoff=10)
    Scores = {qid:top10_BM25}
    BM25_score_10 = {**BM25_score_10 ,**Scores}
    
BM25_score_3 = {}
for qid in qid_query_valid.keys():
    top3_BM25 = BM25(qid,pid_passage_valid,qid_query_valid,inverted_index,k1=1.2,k2=100,b=0.75,cutoff=3)
    Scores = {qid:top3_BM25}
    BM25_score_3 = {**BM25_score_3 ,**Scores}


def Relevant_dict(data):
    """
        A relevant and irrelevant passage function
        Inputs:
          data: input dataset
        Outputs:
          relevant_dict: relevant passage dictionary with a format of {qid: {pid, position}}
          irrelevant_dict: irrelevant passage dictionary with a format of {qid: {pid, position}}
    """
    qid_list = data.qid
    pid_list = data.pid
    relevancy_list = data.relevancy
    relevant_dict = {}
    irrelevant_dict = {}
    for ind,qid in enumerate(qid_list):
        pid = pid_list[ind]
        relevancy = relevancy_list[ind]
        if relevancy > 0:
            if qid not in relevant_dict.keys():
                relevant_dict[qid] = {pid:ind}
            elif qid in relevant_dict.keys():
                new_pid = {pid:ind}
                relevant_dict[qid].update(new_pid)
        else:
            if qid not in irrelevant_dict.keys():
                irrelevant_dict[qid] = {pid:ind}
            elif qid in irrelevant_dict.keys():
                new_pid = {pid:ind}
                irrelevant_dict[qid].update(new_pid)

    return relevant_dict,irrelevant_dict

valid_relevant_dict, valid_irrelevant_dict = Relevant_dict(validation_data)
# 1148 qids
# valid_relevant_dict[1082792]  # given qid, return relevant pids and positions in validation_data
# valid_irrelevant_dict[1082792] # given qid, return irrelevant pids and positions in validation_data
# len(valid_irrelevant_dict[995825])
# validation_data[ind:ind+1]

def mean_AP(model):
    """
        A mean average precision function
        Inputs:
          model: retrivel systems
        Outputs:
           np.mean(AP): mean average precision
           AP: a list containing APs for each query
    """
    AP = []
    qid_list = model.keys()
    
    # for each query
    for qid in qid_list: 
        # find all the relevant pid for this qid
        rel_pid = valid_relevant_dict[qid]
        
        # if the model does not retrieve any relevant pid, Precision is 0
        fail_to_find = [False for i in rel_pid.keys() if i not in model[qid].keys()]
        if fail_to_find:
            metric = 0
        
        # if the model retrieves relevant pids
        else:
            # RR      Relevant Retrieved
            # TR      Current Ranking
            RR = 0
            TR = 0
            Precision = 0
            
            # for each pid retrieved for this qid
            for i in model[qid].keys():
                TR += 1
                # if retrieved a relevant pid, RR add 1, Precision = RR/TR * relevancy
                if i in rel_pid.keys():
                    RR += 1
                    rel = float(validation_data[rel_pid[i]:rel_pid[i]+1].relevancy)
                    Precision += RR / TR * rel
                
                # if retrieved a relevant pid, RR add 1, Precision = RR/TR
                else:
                    Precision += 0 
                    
                # if finding all the relevant pids, stop
                if RR == len(rel_pid.keys()):
                    break   
                    
            # average precision
            metric = Precision / RR 
        
        AP.append(metric)
    
    return  np.mean(AP), AP

mAP_100, AP_100 = mean_AP(BM25_score_100)  # 0.23294399130269514
mAP_10, AP_10 = mean_AP(BM25_score_10) #  0.21670538133952771
mAP_3, AP_3 = mean_AP(BM25_score_3) # 0.17668408826945411


def mean_NDCG(model):
    """
        A mean NDCG function
        Inputs:
          model: retrivel systems
        Outputs:
           np.mean(NDCG): mean NDCG
           NDCG: a list containing NDCGs for each query
    """
    NDCG = []
    qid_list = model.keys()
    
    # for each query
    for qid in qid_list:
        # find all the relevant pid for this qid
        rel_pid = valid_relevant_dict[qid]
        
        # if the model does not retrieve any relevant pid, NDCG is 0
        fail_to_find = [False for i in rel_pid.keys() if i not in model[qid].keys()]
        if fail_to_find:
            metric = 0   
            
        # if the model retrieves relevant pids
        else:
            opt_dict = {} # create a list to store the perfect ranking
            
            # for each pid in relevant pid for this qid
            for i in rel_pid.keys():
                DCG = 0
                
                # if the relevant pid is retrieved, DCG = (2^rel-1)/log_2(rank+1)
                if i in model[qid].keys():
                    rel = float(validation_data[rel_pid[i]:rel_pid[i]+1].relevancy)
                    rank = list(model[qid].keys()).index(i) + 1
                    DCG += (2**rel-1) / np.log2(rank + 1)
                    
                # rerank by optimal ranking
                for ind in rel_pid.values():
                    opt = {i:validation_data.relevancy[ind]}
                    opt_dict = {**opt_dict,**opt} 
                Opt_dict = [(k,opt_dict[k]) for k in sorted(opt_dict.keys(),reverse = True)] 
                
                opt_DCG = 0 
                for i in range(len(Opt_dict)):
                    rel = Opt_dict[i][1]
                    rank = i+1
                    opt_DCG += (2**rel-1) / np.log2(rank + 1)
            
            # NDCG
            metric = DCG / opt_DCG
        
        NDCG.append(metric)
        
    return np.mean(NDCG), NDCG

mNDCG_100, NDCG_100 = mean_NDCG(BM25_score_100) # 0.3440848330645072
mNDCG_10, NDCG_10 = mean_NDCG(BM25_score_10) # 0.27675667124740916
mNDCG_3, NDCG_3 = mean_NDCG(BM25_score_3) # 0.19790956977041063

################################################
## Task 2 Logistic Regression (LR)
################################################

# format : qid pid queries passage relevancy
# 4364339 rows × 5 columns
train_data = pd.read_csv('train_data.tsv', sep='\t',header=0,low_memory=False)

train_relevant_dict, train_irrelevant_dict = Relevant_dict(train_data)
#len(train_relevant_dict.keys()) # 4590
#len(train_irrelevant_dict.keys()) # 4589

def subsampling(data):
    """
        A subsampling function
        Inputs:
          data: input dataset
        Outputs:
           dataset after negative down sampling
    """
    # a list store all subsamples' positions selected
    DF_list = []
    
    # for each query
    for qid in train_relevant_dict.keys():   
        
        # keep all relevant passage, record their positions
        rel_list = list(train_relevant_dict[qid].values())
        
        # random choose samples from irrelevant passage with a rate of 0.025, 
        # record their positions
        if qid not in train_irrelevant_dict.keys():
            irrel_list = []
            
        else:
            L = list(train_irrelevant_dict[qid].values())
            
            # if the number of irrelevant passages for this qid is samller than 25, 
            # keep all irrelevant passages
            if len(L) <= 25:
                irrel_list = L
                
            # if the number of irrelevant passages for this qid is larger than 25,
            # choose them by the rate of 0.025
            else:
                irrel_list = random.sample(L,25) 
                # choose 25 here, since most amount of irrelevant passages is around 1000
                # 1000*0.025 = 25
        
        sample_ind = rel_list + irrel_list
        DF_list += sample_ind  
    
    # convert positions to their corresponding rows
    NewData = []
    for i in DF_list:
        newdata = data[i:i+1]
        NewData.append(newdata)
    
    # merge all the subsamples and convert to a dataFrame
    return pd.concat(NewData,axis=0,ignore_index=True)

# subsampling training dataset
train_subdata = subsampling(train_data)
train_subdata # 118434 rows
subtrain_relevant_dict, subtrain_irrelevant_dict = Relevant_dict(train_subdata)
#len(subtrain_relevant_dict.keys())   # 4590 qids
#len(subtrain_irrelevant_dict.keys())  # 4589 qids


# 118434 rows
clean_train_passage = preprocessing(train_subdata.passage, stopword_removal = True, lemma = True)
clean_train_query = preprocessing(train_subdata.queries, stopword_removal = True, lemma = True)

# 4590 qids and queries
qid_query_train = dict(zip(train_subdata.qid,clean_train_query))

# 115475 pids and passage
pid_passage_train = dict(zip(train_subdata.pid,clean_train_passage))

# saving files
#np.save('qid_query_train.npy',qid_query_valid)
#np.save('pid_passage_train.npy',pid_passage_valid)
#qid_query_train = np.load('qid_query_train.npy',allow_pickle=True).tolist()
#pid_passage_train = np.load('pid_passage_train.npy',allow_pickle=True).tolist()


# word emebedding for all text
# word emebedding for passage in training data
with open('clean_train_passage.txt','w') as f:
    for i in range(len(clean_train_passage)):
        f.write(' '.join(clean_train_passage[i])+'\n')
        
sentences = LineSentence('clean_train_passage.txt')
model_train_passage = Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=1,negative=5,hs=0, workers=4)

# word emebedding for query in training data
with open('clean_train_query.txt','w') as f:
    for i in range(len(clean_train_query)):
        f.write(' '.join(clean_train_query[i])+'\n')

sentences = LineSentence('clean_train_query.txt')
model_train_query = Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=1,negative=5,hs=0, workers=4)

# word emebedding for passage in test data
with open('clean_validation_passage.txt','w') as f:
    for i in range(len(clean_validation_passage)):
        f.write(' '.join(clean_validation_passage[i])+'\n')

sentences = LineSentence('clean_validation_passage.txt')
model_valid_passage = Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=1,negative=5, hs=0, workers=4)

# word emebedding for query in test data
with open('clean_validation_query.txt','w') as f:
    for i in range(len(clean_validation_query)):
        f.write(' '.join(clean_validation_query[i])+'\n')

sentences = LineSentence('clean_validation_query.txt')
model_valid_query = Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=1,negative=5, hs=0, workers=4)


def average_embedding(data,model):
    """
        A average embedding function
        Inputs:
          data: input dictionary with format of {ind: query/passage}
          model: word vector model
        Outputs:
           EM: dictionary with a format of {ind: average embedding for query/passage}
    """    
    EM = {} # empty dictionary
    
    # for each index in dataset
    for i in data.keys():
        text = data[i] # corresponding passage/query
        n = len(text) # length of passage/query
        
        # if the passage/query is not empty
        if n != 0:
            # sum the word vectors for each token in passage/query and then average them
            text_vector = sum(model.wv[text])/n 
                
            # store in a dictionary
            em = {i:text_vector}
            EM = {**EM ,**em}  
        
    return EM


embedding_train_query = average_embedding(qid_query_train, model_train_query) # 4589
embedding_train_passage = average_embedding(pid_passage_train, model_train_passage) # 115475
embedding_valid_query = average_embedding(qid_query_valid, model_valid_query) # 1148
embedding_valid_passage = average_embedding(pid_passage_valid, model_valid_passage) # 955211

# saving word vectors
#np.save('embedding_train_query.npy',embedding_train_query)
#np.save('embedding_train_passage.npy',embedding_train_passage)
#np.save('embedding_valid_query.npy',embedding_valid_query)
#np.save('embedding_valid_passage.npy',embedding_valid_passage)


def new_dataset(data, embedding_query, embedding_passage, relevant_dict):
    """
        A function construct new dataFrame with format <qid, pid, query vector, passage vector, relevancy>
        Inputs:
          data: original dataset 
          embedding_query: embeddings for queries in data
          embedding_passage: embeddings for passages in data
          relevant_dict: relevancy in data
        Outputs:
           new_dataFrame: same as original dataset with query and passage columns changing to embedding vectors
    """
    QID = []
    PID = []
    Query = []
    Passage = []
    Rel = []
    for i in range(len(data)):
        qid = data.qid[i]
        pid = data.pid[i]
        if qid in embedding_query.keys() and pid in embedding_passage.keys():
            query_vec = embedding_query[qid]
            passage_vec = embedding_passage[pid]
            QID.append(qid)
            PID.append(pid)
            Query.append(query_vec)
            Passage.append(passage_vec)
            if pid in relevant_dict[qid].keys():
                i = relevant_dict[qid][pid]
                rel = float(data[i:i+1].relevancy)
            else:
                rel = 0
            Rel.append(rel)
            
    new_data = list(zip(QID,PID,Query,Passage,Rel)) 
    new_dataFrame = pd.DataFrame(data = new_data, columns=['qid','pid','q_vec','p_vec','relevancy'])
    
    return new_dataFrame

## Construct Train dataset
########################################
# word vector datset
train_wv = new_dataset(train_subdata, embedding_train_query, embedding_train_passage, subtrain_relevant_dict)
train_wv # 118425 rows

valid_wv = new_dataset(validation_data, embedding_valid_query, embedding_valid_passage, valid_relevant_dict)
valid_wv # 1103039 rows

train_wv.insert(2,'One',1)
dataset_train = train_wv.values # convert dateFrame to matrix

def sigmoid(z):
    """
        A sigmoid function
    """
    return 1 / (1 + np.exp(-z))

def model_func(X,w):
    """
        A  logistic function 
    """
    return sigmoid(np.dot(X,w.T))

def pred_func(w, X):
    """
       A  predict function 
    """
    return model_func(X,w)

def loss_func(X,y,w):
    """
       A  loss function 
    """   
    sigma = model_func(X,w)
    loss = -np.sum(np.multiply(y,np.log(sigma))+np.multiply(1-y,np.log(1 - sigma)))/len(X)
    return loss
    
def grad_func(X,y,w):
    """
       Partial derivative of loss function 
    """       
    n, d = X.shape
    grad = np.zeros((d))
    diff = [a - b for a,b in zip(y, model_func(X,w))] 
    for j in range(d):
        x_ij = X[:,j]
        grad[j] = -np.sum(np.multiply(x_ij,diff))/n
    return grad

def shuffleData(dataset):
    """
       Shuffle dataset and separate data inputs and labels
       Outputs:
          X_train: data inputs
          y_train: data labels
    """   
    np.random.shuffle(dataset)
    X_inputs = dataset[:,2:5]
    X_train = []
    for i in range(len(X_inputs)):
        x = np.hstack((X_inputs[i][0], X_inputs[i][1],X_inputs[i][2]))
        X_train.append(x)
        
    X_train = np.array(X_train) # (118425, 201)
    y_train = np.array(dataset[:,5:6]) # (118425, 1)
    
    return X_train,y_train

def grad_descent(data, step_size, max_iter, lr, tol):
    """
        Mini batch stochastic gradient descent
        Inputs:
          data: training dataset
          step_size: batch size 
          max_iter: max iterations
          lr: learning rate
          tol: tolerance
        Outputs:
           new_dataFrame: same as original dataset with query and passage columns changing to embedding vectors
    """
    iteration = 0
    step = 0
    X,y = shuffleData(data)
    w = np.zeros([1,201]) # inital weight 
    grad = np.zeros(len(w))
    Loss = []
    stop_Criterion = True
    
    while (iteration < max_iter and stop_Criterion == True):
        loss = loss_func(X,y,w)
        grad = grad_func(X[step:step+step_size], y[step:step+step_size], w)
        w -= lr*grad
        step += step_size
        if step >= len(X):
            step = 0
            X,y = shuffleData(data)
        #for i in range(len(w)):
        #    w[i] -= lr*grad[i]
        Loss.append(loss)
        iteration += 1       
        
        # stop_Criterion
        if np.linalg.norm(grad) < tol:
            stop_Criterion == False
            
    return w, Loss, iteration-1

def logistic_experiment_lr(data,step_size,max_iter,lr,tol):
    """
        Analyze the effect of the learning rate on the model training loss.
        Inputs:
          data: training dataset
          step_size: batch size 
          max_iter: max iterations
          lr: learning rate list
          tol: tolerance
        Outputs:
          plot losses vs iterations for each learning rate 
    """
    Losses =[]
    for r in lr:
        weight, Loss, iterations = grad_descent(data, step_size, max_iter, r, tol)
        Losses.append(Loss)
    plt.figure(figsize=(8,6))
    plt.xlabel('Iterations', fontsize=17)
    plt.ylabel('Training Loss', fontsize=17)
    plt.title('Training loss for different learning rates', fontsize=17)
    for i in range(len(Losses)):
        plt.plot(np.arange(len(Losses[i])),Losses[i],label='lr ='+ str(lr[i]))
    
    plt.legend()
    plt.savefig('lr.pdf', bbox_inches = 'tight')

lr = [0.01,0.005,0.001,0.0005,0.0001]
logistic_experiment_lr(dataset_train,step_size=10, max_iter =5000,lr=lr,tol=0.01)
Weight,_,I = grad_descent(dataset_train, step_size=10, max_iter=10000, lr = 0.01, tol=0.001)

def LR_predict(qid, embedding_query, embedding_passage, W, cutoff=100):
    """
        Calculate the predicted score with given qid for each pid
        Inputs:
          qid: query ID
          embedding_query: embedding vectors for query 
          embedding_passage: embedding vectors for passage 
          W: weight obtrain from training
          cutoff: choose ranking top 'k' 
        Outputs:
          top_score: given qid, return its top 'k' pids and scores
    """
    pids,_ = qid_to_pid(qid, qid_list, pid_list)
    q_vec = embedding_query[qid]
    score_list = {}
    for pid in pids:
        p_vec = embedding_passage[pid]
        word_vec = np.hstack((1,q_vec,p_vec))
        score = pred_func(W, word_vec)
        Scores = {pid:score}
        score_list = {**score_list ,**Scores}
    top_score = dict(sorted(score_list.items(), key=lambda x: x[1], reverse=True)[:cutoff])
    return top_score

LR_score = {}
for qid in qid_query_valid.keys():
    top_score_LR = LR_predict(qid, embedding_valid_query, embedding_valid_passage, W=Weight, cutoff=100)
    if len(top_score_LR) == 100:
        Scores = {qid:top_score_LR}
        LR_score = {**LR_score ,**Scores}

mAP_LR, AP_LR = mean_AP(LR_score) # 0.004042322132835292
mNDCG_LR, NDCG_LR = mean_NDCG(LR_score) # 0.020584012397022223

with open('LR.txt','w') as f:
    for i in range(len(LR_score.keys())):
        qid = list(LR_score.keys())[i]
        pids = list(LR_score[qid].keys())
        for j in range(100):
            pid = pids[j]
            # qid A2 pid rank score algoname
            f.writelines([str(qid), '  A2  ', str(pid),'  ', str(j+1),'  ',str(float(LR_score[qid][pid])), '  LR', '\n'])
f.close()

################################################
## Task 3 LambdaMART Model (LM)
################################################
def group_split_train(qid_list):
    Group = []
    qids = set(qid_list)
    for qid in qids:
        num = qid_list.count(qid)
        Group.append(num)
    return Group

group_train = group_split_train(list(train_wv.qid))

models = []
Lr = [0.01,0.005,0.001]
Estimator = [100,200,300]
Depth = [5,6,7]

for i in range(len(Lr)):
    for j in range(len(Estimator)):
        for k in range(len(Depth)):
            models.append(xgb.XGBRanker(  
                           booster='gbtree',
                           objective='rank:pairwise',
                           eta=Lr[i],
                           max_depth=Depth[k], 
                           n_estimators=Estimator[j]
                           ))

train_labels = train_wv.values[:,5:6] # (118425, 1)
X_inputs = train_wv.values[:,3:5]
X_train = []
for i in range(len(X_inputs)):
    x = np.hstack((X_inputs[i][0], X_inputs[i][1]))
    X_train.append(x)
train_inputs = np.array(X_train) # (118425, 200)

LM_models = []
i = 1
for model in models:
    print('model '+ str(i))     
    LM_models.append(model.fit(train_inputs, train_labels, group=group_train, verbose=0))
    i+=1

def LM_predict(qid,model,embedding_query, embedding_passage,cutoff=100):
    """
        Calculate the predicted score with given qid for each pid
        Inputs:
          qid: query ID
          model: LambdaMART Model
          embedding_query: embedding vectors for query 
          embedding_passage: embedding vectors for passage 
          cutoff: choose ranking top 'k' 
        Outputs:
          top_score: given qid, return its top 'k' pids and scores
    """    
    pids,_ =qid_to_pid(qid, qid_list, pid_list)
    q_vec = embedding_query[qid]
    score_list = {}
    group_test = []
    for pid in pids:
        p_vec = embedding_passage[pid]
        test_inputs = np.hstack((q_vec,p_vec))
        group_test.append(test_inputs)
    group_test = np.array(group_test)
    preds = model.predict(group_test)
    
    for i in range(len(pids)):
        score = float(preds[i])
        Scores = {pids[i]:score}
        score_list = {**score_list ,**Scores}
    top_score = dict(sorted(score_list.items(), key=lambda x: x[1], reverse=True)[:cutoff])
    return top_score

LM_scores = []
for i in range(len(LM_models)):
    print('model '+ str(i+1))
    for qid in qid_query_valid.keys():
        top_score_LM = LM_predict(qid,LM_models[i],embedding_valid_query, embedding_valid_passage,cutoff=100)    
        if len(top_score_LM) == 100:
            Scores = {qid:top_score_LM}
            LM_score = {**LM_score ,**Scores}
    LM_scores.append(LM_score)

for i in range(len(LM_scores)):
    mAP_LM,_ = mean_AP(LM_scores[i])
    print('mAP of model '+ str(i+1)+' is ' + str(mAP_LM))

for i in range(len(LM_scores)):
    mNDCG_LM,_ = mean_NDCG(LM_scores[i])
    print('mNDCG of model '+ str(i+1)+' is ' + str(mNDCG_LM))

LM_score = LM_scores[11]
with open('LM.txt','w') as f:
    for i in range(len(LM_score.keys())):
        qid = list(LM_score.keys())[i]
        pids = list(LM_score[qid].keys())
        for j in range(100):
            pid = pids[j]
            # qid A2 pid rank score algoname
            f.writelines([str(qid), '  A2  ', str(pid),'  ', str(j+1),'  ',str(float(LM_score[qid][pid])), '  LM', '\n'])
f.close()

################################################
## Task 4 Neural Network Model (NN)
################################################
## restart the kernel

# format : qid pid queries passage relevancy
# 1103039 rows × 5 columns
validation_data = pd.read_csv('validation_data.tsv', sep='\t',header=0,low_memory=False)

# 4364339 rows × 5 columns
train_data = pd.read_csv('train_data.tsv', sep='\t',header=0,low_memory=False)

def preprocessing(text, stopword_removal = False, lemma = False):
    """
        A text preprocessing function
        Inputs:
          text: input queries/passages
          stopword_removal: remove all stopwords if True
          lemma: do lemmatisation and stemming if True
        Outputs:
          passage: queries/passages after preprocessing
    """
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

def Relevant_dict(data):
    """
        A relevant and irrelevant passage function
        Inputs:
          data: input dataset
        Outputs:
          relevant_dict: relevant passage dictionary with a format of {qid: {pid, position}}
          irrelevant_dict: irrelevant passage dictionary with a format of {qid: {pid, position}}
    """
    qid_list = data.qid
    pid_list = data.pid
    relevancy_list = data.relevancy
    relevant_dict = {}
    irrelevant_dict = {}
    for ind,qid in enumerate(qid_list):
        pid = pid_list[ind]
        relevancy = relevancy_list[ind]
        if relevancy > 0:
            if qid not in relevant_dict.keys():
                relevant_dict[qid] = {pid:ind}
            elif qid in relevant_dict.keys():
                new_pid = {pid:ind}
                relevant_dict[qid].update(new_pid)
        else:
            if qid not in irrelevant_dict.keys():
                irrelevant_dict[qid] = {pid:ind}
            elif qid in irrelevant_dict.keys():
                new_pid = {pid:ind}
                irrelevant_dict[qid].update(new_pid)

    return relevant_dict,irrelevant_dict

valid_relevant_dict, valid_irrelevant_dict = Relevant_dict(validation_data)
train_relevant_dict, train_irrelevant_dict = Relevant_dict(train_data)

def subsampling(data):
    """
        A subsampling function
        Inputs:
          data: input dataset
        Outputs:
           dataset after negative down sampling
    """
    # a list store all subsamples' positions selected
    DF_list = []
    
    # for each query
    for qid in train_relevant_dict.keys():   
        
        # keep all relevant passage, record their positions
        rel_list = list(train_relevant_dict[qid].values())
        
        # random choose samples from irrelevant passage with a rate of 0.025, 
        # record their positions
        if qid not in train_irrelevant_dict.keys():
            irrel_list = []
            
        else:
            L = list(train_irrelevant_dict[qid].values())
            
            # if the number of irrelevant passages for this qid is samller than 25, 
            # keep all irrelevant passages
            if len(L) <= 5:
                irrel_list = L
                
            # if the number of irrelevant passages for this qid is larger than 25,
            # choose them by the rate of 0.025
            else:
                irrel_list = random.sample(L,5) 
                # choose 25 here, since most amount of irrelevant passages is around 1000
                # 1000*0.025 = 25
        
        sample_ind = rel_list + irrel_list
        DF_list += sample_ind  
    
    # convert positions to their corresponding rows
    NewData = []
    for i in DF_list:
        newdata = data[i:i+1]
        NewData.append(newdata)
    
    # merge all the subsamples and convert to a dataFrame
    return pd.concat(NewData,axis=0,ignore_index=True)

train_subdata = subsampling(train_data)
train_subdata # 27726 rows

subtrain_relevant_dict, subtrain_irrelevant_dict = Relevant_dict(train_subdata)
#len(subtrain_relevant_dict.keys())   # 4590 qids
#len(subtrain_irrelevant_dict.keys())  # 4589 qids

## New embedding
txt = "glove.6B.100d.txt"
 
with open(txt, 'r') as f:
    line = f.readline().split(' ')
    vector_size = len(line) - 1
    
vocab_size = -1
for vocab_size, line in enumerate(open(txt,'rU')):
    pass
vocab_size += 1
    
word_model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False)
len(list(word_model.key_to_index))  # 400000 words, 100 dim for each word


max_length = 0
for i in tqdm(range(len(qid_query_train.keys()))):
    length = len(list(qid_query_train.values())[i])
    if length > max_length:
        max_length = length 
for i in tqdm(range(len(pid_passage_train.keys()))):
    length = len(list(pid_passage_train.values())[i])
    if length > max_length:
        max_length = length
for i in tqdm(range(len(qid_query_valid.keys()))):
    length = len(list(qid_query_valid.values())[i])
    if length > max_length:
        max_length = length
for i in tqdm(range(len(pid_passage_valid.keys()))):
    length = len(L[i])
    if length > max_length:
        max_length = length 
max_length # 137 words

train_passages = preprocessing(train_subdata.passage, stopword_removal = True, lemma = False)
train_queries = preprocessing(train_subdata.queries, stopword_removal = True, lemma = False)
test_passages = preprocessing(validation_data.passage, stopword_removal = True, lemma = False)
test_queries = preprocessing(validation_data.queries, stopword_removal = True, lemma = False)\

def word_table(datasets,model):
    token_to_ind = {} # tokens to indexes
    ind_to_vec = {} # indexes to word vectors
    i = 0
    
    for dataset in tqdm(datasets):
        for sentence in dataset: # for each query/passage
            for token in sentence: # for each token of the sentence
                # if this word is not token_to_ind
                if(token_to_ind.get(token) == None):
                    if token in model:
                    # if this word exists is the word model
                        i += 1
                        token_to_ind[token] = i
                        ind_to_vec[i] = model[token]

    return token_to_ind, ind_to_vec

# 150119 words
token_ind_dict, ind_vec_dict = word_table([train_passages,train_queries,test_passages,test_queries], word_model)


def new_sentence_embedding(token_ind_dict,ind_vec_dict,text):
    sentence_vec = []
  ## for every sentence
    for sentence in tqdm(text):
        sentence_vec_list = []
        for word in sentence:
            word_index = token_ind_dict.get(word)
        if(word_index!=None):
            word_embedding = ind_vec_dict.get(word_index)
            sentence_vec_list.append(np.array(word_embedding))
        else:
            sentence_vec_list.append(np.array(np.zeros(100)))
    if len(sentence_vec_list) == 0:
        sentence_vec_list.append(np.array(np.zeros(100)))
        sentence_vec_list = np.array(sentence_vec_list)

    sentence_vec.append(np.mean(sentence_vec_list,axis = 0))

    return np.array(sentence_vec)


train_passage_ind = new_sentence_embedding(token_ind_dict, ind_vec_dict, train_passages)
train_queries_ind= new_sentence_embedding(token_ind_dict, ind_vec_dict, train_queries)
test_passage_ind = new_sentence_embedding(token_ind_dict, ind_vec_dict, test_passages)
test_queries_ind = new_sentence_embedding(token_ind_dict, ind_vec_dict, test_queries)
train_labels = train_subdata['relevancy'].values
test_labels = validation_data['relevancy'].values

def look_up_table(ind_vec_dict):
    table = [np.zeros(100)]
    for key in sorted (ind_vec_dict.keys()) :  
        table.append(ind_vec_dict.get(key))
    return np.array(table)

new_table = look_up_table(ind_vec_dict)
print(new_table.shape) # (150120, 100)


def padding_zeros(max_length,vector):
    vector = np.array(vector)
    if(vector.shape[0] < max_length and vector.shape[0] != 0):
       padding_vector = np.zeros(max_length - vector.shape[0])
       return np.concatenate((vector, padding_vector), axis=0)
    elif(vector.shape[0] == max_length):
       return vector
    else:
       return np.zeros((0,0))


def word_to_index_func(text,max_length,labels,token_ind_dict):
    embedding_ind = []
    embedding_labels = []
    sentence_lengths = []
    i = -1

    for sentence in tqdm(text):
        i += 1
        embedding_sentence = []
        for word in sentence:
            if(token_ind_dict.get(word)!=None):
                embedding_sentence.append(token_ind_dict.get(word))

    sentence_lengths.append(len(embedding_sentence))

    embedding_sentence =  padding_zeros(max_length,embedding_sentence) 
    embedding_labels.append(labels[i])
    
    if(embedding_sentence.shape[0]!=0):
        embedding_ind.append(np.array(embedding_sentence))
    else:
        embedding_ind.append(np.zeros(max_length))
    Ind = np.array(embedding_ind)
    Labels = np.array(embedding_labels)
    Len = np.array(sentence_lengths)
    return Ind,Labels,Len


train_pInd,train_labels,train_plen = word_to_index_func(train_passages,max_length,train_labels,token_ind_dict)
train_qInd,_,train_qlen = word_to_index_func(train_queries,max_length,train_labels,token_ind_dict)

test_pInd,test_labels,test_plen = word_to_index_func(test_passages,max_length,test_labels,token_ind_dict)
test_qInd,_,test_qlen = word_to_index_func(test_queries,max_length,test_labels,token_ind_dict)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_pInd = torch.from_numpy(train_pInd).int().to(device)
train_qInd = torch.from_numpy(train_qInd).int().to(device)
train_plen = torch.from_numpy(train_plen)
train_plen = torch.as_tensor(train_plen,dtype=torch.int64)
train_qlen = torch.from_numpy(train_qlen)
train_qlen = torch.as_tensor(train_qlen, dtype=torch.int64)
train_labels = torch.from_numpy(train_labels).to(device)

new_table = torch.from_numpy(new_table).float().to(device)

test_pInd = torch.from_numpy(test_pInd).int().to(device)
test_qInd = torch.from_numpy(test_qInd).int().to(device)
test_plen = torch.from_numpy(test_plen)
test_plen = torch.as_tensor(test_plen, dtype=torch.int64)
test_qlen = torch.from_numpy(test_qlen)
test_qlen = torch.as_tensor(test_qlen, dtype=torch.int64)
test_labels = torch.from_numpy(test_labels).to(device)

train_data = torch.utils.data.TensorDataset(train_pInd, train_plen, train_qInd, train_qlen, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

test_data = torch.utils.data.TensorDataset(test_pInd, test_plen, test_qInd, test_queries_lengths, test_qlen)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

class RNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(RNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = 2,
                          bidirectional = True, dropout = 0.5)
        self.fc = nn.Linear(hidden_dim*2,1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x):
        embedding = self.dropout(self,embedding(x))
        output,(hidden,cell) = self.rnn(embedding)
        hidden - torch.cat(hidden[-2],hidden[-1],dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out

    
