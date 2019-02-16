#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time

### features_and labels of train and test
features_train, features_test, labels_train, labels_test = preprocess()

#Reduce in 99% the dataset
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#creating model
clf = SVC(kernel='rbf', C=10000)

#time trainning
t0 = time()

#fitting the model
clf.fit(features_train, labels_train)

print "Tempo de treinamento:", round(time()-t0, 3), "s"

#time predict
t1 = time()

#predict
pred = clf.predict(features_test)

print "Tempo de predicao:", round(time()-t1, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print'Acuracia:', accuracy

print 'Total: ', pred.shape[0]
print 'Classifiers of Chris: ', pred[pred == 1].shape[0]