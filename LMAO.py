import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from collections import Counter
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

"""Userhandbook
1) Complicated & uncommon words mess it up
2) Don't input titles that are too short
3) Perform better with jokes from 7~65 words
4) Potential bugs when there's jump in content within a joke
5) This is a binary prediction, so the absolute value is just for reference
6) No emoji & stuff
"""

# load model
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(20000, 256) #Consider based on frequency
        self.cnn1 = nn.Sequential( # batch, embeddingdim, vacablen CNN
            nn.Conv1d(256, 512, 4, stride=2), #As indicated by stanford essay. 7, 5, 3, 1:1 ((x-1)/2)
            nn.BatchNorm1d(512), # span (limit) large numbers to -1 ~ 1
            nn.ReLU(), # negative value goes to 0, throw away redundent information
            nn.Dropout(p=0.2), # reducing overfitting by randomly setting a fraction of input units to zero during training. Common value between 0.2 & 0.5, smaller model & larger dataset drop off rate --> 0.2. Needs fine tune
            nn.Conv1d(512, 128, 4, stride=2),
            nn.BatchNorm1d(128), # span (limit) large numbers to -1 ~ 1
            nn.ReLU(), # negative value goes to 0, throw away redundent information
        )
        self.cnn2 = nn.Sequential( # batch, embeddingdim, vacablen
            nn.Conv1d(256, 512, 7, stride=3),#As indicated by stanford essay
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 128, 7, stride=3),#As indicated by stanford essay
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )


        self.output_layer = nn.Sequential( #out put to 1 Fully connect neural nerwork
            nn.Linear(2432,1024), # power of 2. (in_features, outfeatures)In natural language processing tasks with word embeddings, in_features would be the dimensionality of the word embeddings. Do I need to experiment with it?
            nn.ReLU(), #Non linearity
            nn.Dropout(0.2), #Overfitting
            nn.Linear(1024,1024),# *2
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,1),# goes to 4 classes
            nn.Sigmoid()
            # nn.Sigmoid() #Important, limitations
        )

    def forward(self, src): #combine
        src = self.embedding(src).transpose(1,2) #embed, transpose the domentions of output sensor(why?)
        x1 = self.cnn1(src) #apply to layer, extract interesting input
        x2 = self.cnn2(src)

        logits = torch.cat((x1, x2), 2)# output of two layers are concatenated along featured dimension, extract releven features
        # print(logits.shape)
        logits = self.output_layer(logits.flatten(1)) #from concatenates vector to 2 dimentional outpu

        return logits

model = Model()

param = torch.load("model.pth.tar")["state_dict"]
param["embedding.weight"][0] = param["embedding.weight"][1]

model.load_state_dict(param)
model = model.eval()

#Tokenizer
class Tokenizer:
    def __init__(self):
        # two special tokens for padding and unknown, Exception: when you wanna seperate the text
        self.token2idx = {"<pad>": 0, "<unk>": 1} #Word exceeding limit, padding: special symbols --> you usually want to include it
        self.idx2token = ["<pad>", "<unk>"]
        self.is_fit = False 
    
    @property
    def pad_id(self):
        return self.token2idx["<pad>"]
    
    def __len__(self):
        return len(self.idx2token)
    
    def fit(self, train_texts: List[str]):
        counter = Counter()
        for text in tqdm(train_texts):
            counter.update(text.lower().split())
        
        # manually set a vocabulary size for the data set
        vocab_size = 20000
        self.idx2token.extend([token for token, count in counter.most_common(vocab_size - 2)])
        for (i, token) in enumerate(self.idx2token):
            self.token2idx[token] = i
            
        self.is_fit = True
                
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        if not self.is_fit:
            raise Exception("Please fit the tokenizer on the training tokens")
        
        tokens = text.lower().split()
        token_ids = [self.token2idx.get(token, self.token2idx["<unk>"]) for token in tokens]

        if max_length is not None:
            # truncate or pad the token_ids to max_length
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids += [self.token2idx["<pad>"]] * (max_length - len(token_ids))

        return token_ids

'''
#Data Cleansing
dataarr = []
with open("fullrjokes.json", 'r') as f:
    for line in f:
        dataarr.append(json.loads(line))
        
data = pd.DataFrame(dataarr)
data["selftext"] = data["selftext"].map(lambda x : "" if x in ["[deleted]","[removed]"] else x + " ") #Need to make sure words are seperated.
data = data.dropna()
data["joke"] = data["selftext"] + data["title"]
order = ['joke', 'downs', 'ups', 'score']
data = data[order]
data = data.reset_index()
data['joke'] = data['joke'].str.replace('\n', ' ')
data['joke'] = data['joke'].str.replace('\r', ' ')


re.split("[,.;:?!&\\(\\)\"] *", "Everybody likes dark humor. What's the differe") #syntax is fixed, can't replace ' because end up in lots of s

data['joke'] = data['joke'].map(lambda x : re.sub("[,.;:?!&\\(\\)\"] *", '', x))
data['joke']

from collections import Counter
dataCount = Counter()

data["joke"].map(lambda x : dataCount.update(x.lower().split(" ")))

data["pScore"] = data["score"].map(lambda x : x if x == 0 else np.log(x))
data

data = data.loc[data["pScore"] > 0].copy()

#Tokenize dataset
tokenizer = Tokenizer()
tokenizer.fit(data["joke"])
'''

import pickle

with open("Tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

flag = ""
print("Heyyy, u think u r funny, huh?")
print("Let's test it out.")
input_string = input("Write your joke here: ")

while input_string != "e":
    input_string = re.sub("[,.;:?!&\\(\\)\"“”] *", ' ', input_string)
    print(input_string)
    joke = tokenizer.encode(input_string, max_length=65)
    joke = torch.tensor(joke).reshape(1,-1)
    #print(joke)
    with torch.no_grad():
        output = model(joke)
    print(output[0][0].item())
    input_string = input("Write your joke here. If exit, type e: ")






