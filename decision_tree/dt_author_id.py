#!/usr/bin/python
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


### features_and labels of train and test
features_train, features_test, labels_train, labels_test = preprocess()

#creating model
clf = tree.DecisionTreeClassifier(min_samples_split=40)

#time train
t0 = time()

#fitting model
clf.fit(features_train, labels_train)

print "Tempo de treinamento:", round(time()-t0, 3), "s"

#time test
t1 = time()

#predict
pred = clf.predict(features_test)

print "Tempo de teste:", round(time()-t1, 3), "s"

#accuracy
accuracy = accuracy_score(pred, labels_test)

print 'Acuracia: ', accuracy
print 'Num Atributos: ',len(features_train[0])