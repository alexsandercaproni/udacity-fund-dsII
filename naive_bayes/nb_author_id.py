#!/usr/bin/python

sys.path.append("../tools/")
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time
from email_preprocess import preprocess



### features and labels of_train and test
features_train, features_test, labels_train, labels_test = preprocess()

#Creating classifier
clf = GaussianNB()

#time trainning
t0 = time()

#Treinando o modelo
clf.fit(features_train, labels_train)

print "Tempo de treinamento:", round(time()-t0, 3), "s"

#time predict
t1 = time()

#Fazendo predicts
pred = clf.predict(features_test)

print "Tempo de teste:", round(time()-t1, 3), "s"

print 'Acuraria: ', accuracy_score(pred, labels_test)