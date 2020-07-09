
import warnings
import os
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

import statistics, math
import numpy as np
from numpy import savetxt
import pickle

from transformers import BertTokenizer, BertModel

def read_path():
    sent_path = 'sentence'
    sub_dir = os.listdir(sent_path)
    path = [sent_path+'/'+str(sub)+'/'+str(sub_path) for sub in sub_dir for sub_path in os.listdir(sent_path+'/'+str(sub))]
    return(path)

def read_data(paths):
    train_sent, result_sent = [], []
    for val in paths[:5]: ######################## READ ONLY 5 PATHS ######################
        data = open(val).read().split('\n')
        train_sent.append(data[:-3])
        result_sent.append(data[-2]) # Snce last value is an empty string
    return train_sent, result_sent


def get_length(S1, S2):
    S1 = sum(S1,[])
    Sent = S1 + S2
    length = [len(val.split(' ')) for val in Sent]
    return length


def get_sent_dim(train_sent, result_sent, max_len):
    train_data = [sent[:max_len] for sent_list in train_sent for sent in sent_list if len(sent)>max_len else sent+['unk']*(max_len - len(sent))] 
    result = [sent[:max_len] for sent_list in result_sent for sent in sent_list if len(sent)>max_len else sent+['unk']*(max_len - len(sent))]
    labels_data = [itertools.chain.from_iterable(itertools.repeat(x, 4) for x in result)]        
    return train_data, labels_data 

def BERT_emb(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = torch.tensor([tokenizer.encode(text, add_special_tokens = True)])
    model = BertModel.from_pretrained("bert-base-uncased")
    embeddings = model(tokens)[0]
    return embeddings 


def Glove_emb(text):
    words = []
    idx = 0
    word_to_idx = {}
    glove_path = os.getcwd()
    vectors = []
    file_ = open(f'{glove_path}/glove.6B.50d.txt', 'rb').readlines()
    for l in file_[:300]:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word_to_idx[word] = idx
        idx += 1
        vect = line[1:]
        vectors += vect
    np_vectors = np.reshape(np.asarray(vectors, dtype=np.float), (-1,50), 'C')
    pickle.dump(np_vectors, open(f'{glove_path}/6B.50_data.pkl', 'wb'))
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word_to_idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))
    #print('The string : {} has vector : {}'.format(words[0], np_vectors[0]))
    vectors = pickle.load(open(f'{glove_path}/6B.50_data.pkl', 'rb'))
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    tokens = text.lower().split()
    glove_tokens = torch.tensor([glove[val] for val in tokens], dtype = torch.long)
    return glove_tokens

def check(G,B,sent_len):
    n = int(np.ceil(B.shape[2]/sent_len))
    B_new = torch.zeros(9,n*sent_len)
    B_new[:,:B.shape[2]] = B[0]
    print('Glove : {}'.format(G.shape))
    print('BERT : {}'.format(B_new.shape))
    return B_new, G

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(770, 128)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(128, 50)
            self.fc3 = torch.nn.Linear(50, 10)
            self.sigmoid = torch.nn.Sigmoid()        

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.relu(self.fc3(output))
            output = self.sigmoid(output)
            return output

def forward(X,Y):
    model = Feedforward(128,128) #(input, hidden)
    val = model(X)
    return val

def main():
    text = "Top university members found help in history"
    paths = read_path()
    train_sent, result_sent = read_data(paths)
       
    len_ = get_length(train_sent, result_sent)
    max_len = math.ceil(statistics.mean(len_) + max(len_)/len(len_))
    train, labels = get_sent_dim(train_sent, result_sent, max_len)
    
    """
    B = BERT_emb(text)
    G = Glove_emb(text)
    X, Y = check(G,B,G.shape[0])
    X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)
    output = forward(X, Y)
    print(output.shape)
    """

if __name__ == "__main__":
    main()


