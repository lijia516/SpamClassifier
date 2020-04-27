#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 01:43:02 2020

@author: lijialiu
"""

###load data####
import pandas as pd
data = pd.read_csv("spam.csv", encoding = "latin-1")
data = data[['v1', 'v2']]
data = data.rename(columns = {'v1': 'label', 'v2': 'text'})
data = data.replace(to_replace =["ham", "spam"], value =[0, 1])


#####set trainind data and testing data########
import numpy as np
def train_test_split(data, rate):
    totalTextNumber = data.shape[0]
    trainIndex = []
    testIndex = []

    for i in range(totalTextNumber):
        if(np.random.uniform(0, 1) < rate):
            trainIndex += [i]
        else:
            testIndex += [i]
    trainText = data.loc[trainIndex]
    testText = data.loc[testIndex]
    return trainText, testText






trainText, testText = train_test_split(data, 0.75)
trainText.reset_index(inplace = True)
trainText.drop(['index'], axis = 1, inplace = True)
#trainText


####precess data#######
import nltk
nltk.download('stopwords')

from nltk import stem
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def stemming(words):
    stemmer = stem.SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    return words
def remove_stop_words(words):
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]
    return words
def lower_case(text):
    text = text.lower()
    return text
def pre_process(text):
    text = lower_case(text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if len(word) > 1]
    words = stemming(words)
    words = remove_stop_words(words)
    return words



######calssifier########
class spamCalssifier(object):
    def __init__(self, trainText):
        self.text = trainText['text']
        self.label = trainText['label']

    def train(self):
        self.tf_and_idf()  # compute the values needed for tf_idf function
        self.tf_idf()
        
    def tf_and_idf(self):
        self.ham_text = self.label.value_counts()[0]
        self.spam_text = self.label.value_counts()[1]
        self.total_texts = self.label.shape[0]
        self.total_ham = 0    #total words in ham texts
        self.total_spam = 0   #total words in spam texts
        self.tf_ham = {}
        self.tf_spam = {}
        self.idf_countText = {}   # the number of messages containing w
        
        for i in range(self.total_texts):
            
            #print(self.text[i])
            
            processed_words = pre_process(self.text[i])
            
            #print(processed_words)
            #exit()
            
            contain_words = set()
            for word in processed_words:
                contain_words.add(word)
                if self.label[i] == 1:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.total_spam += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.total_ham += 1
        
            for word in contain_words:
                    self.idf_countText[word] = self.idf_countText.get(word, 0) + 1
    
    def tf_idf(self):
        self.prob_spam = {}   # p(w|spam) = tf_spam[w] * idf(w) / (summaton of p(w|spam) for all w)
        self.prob_ham = {}    # p(w|ham) 
        self.idf = {}        # idf(w) = log(the number of texts / the number of texts containing w)
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.idf[word] = np.log(self.total_texts / self.idf_countText[word])
            self.prob_spam[word] = self.tf_spam[word]* self.idf[word]
            self.sum_tf_idf_spam += self.prob_spam[word]
            
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(self.prob_spam))
        
        for word in self.tf_ham:
            self.idf[word] = np.log(self.total_texts / self.idf_countText[word])
            self.prob_ham[word] = self.tf_ham[word]* self.idf[word]
            self.sum_tf_idf_ham += self.prob_ham[word]
            
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(self.prob_ham))
        self.prob_spam_text = self.spam_text / self.total_texts
        self.prob_ham_text = self.ham_text / self.total_texts
        
        
    def classify(self, text):
        prob_spamText = 0
        prob_hamText = 0
        
        for word in text:
            if word in self.prob_spam:
                prob_spamText += np.log(self.prob_spam[word])
            else:
                prob_spamText -= np.log(self.sum_tf_idf_spam + len(self.prob_spam))
                
            if word in self.prob_ham:
                prob_hamText += np.log(self.prob_ham[word])
            else:
                prob_hamText -= np.log(self.sum_tf_idf_ham + len(self.prob_ham))
                
        prob_spamText -= np.log(self.prob_spam_text)
        prob_hamText -= np.log(self.prob_ham_text)
        
        return prob_spamText >= prob_hamText
    
    def predict_test(self, testText):
        result = {}
        for i in range(testText.shape[0]):
            words = pre_process(testText['text'][i])
            result[i] = int(self.classify(words))
        return result
    
    
    
myC = spamCalssifier(trainText)
myC.train()

testText.reset_index(inplace = True)
testText.drop(['index'], axis = 1, inplace = True)
#testText
result = myC.predict_test(testText) 
predicted_label = list(result.values())
correct = 0
for i in range(len(result)):
        if testText['label'][i] == result[i]:
            correct += 1
print(correct/len(predicted_label)) 