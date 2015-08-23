# -----------------------------------------------------------#
# MultiNomial Naive Bayes- Rotten Tomatoes Data set
#-----------------------------------------------------------#

import sys
import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.metrics import ConfusionMatrix
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn import cross_validation

def create_template(senti_data):
  data=[]
  for line in  senti_data:
     senti_dict={}
     line=line.strip('\n')
     line=line.split('\t')
     senti_dict[line[2]]='True'
     data.append([senti_dict, line[3]])
  return data

def classify_mnb(input_data,t_data):
  training_sample=create_template(input_data)
  classifier = NaiveBayesClassifier.train(training_sample)
  test_sample=create_template(t_data)
  rsets=collections.defaultdict(set)
  psets=collections.defaultdict(set)
  reference=[]
  predicted=[]
  reference_c=[]
  predicted_c=[]

  for i,(attribute,label) in enumerate(test_sample):
      reference.append(label)
      rsets[label].add(i)
      class_predicted= classifier.classify(attribute)
      psets[class_predicted].add(i)
      predicted.append(class_predicted)
      t_feature=list(attribute.keys())
      sum_accuracy=0.0
  cv = cross_validation.KFold(len(training_sample), n_folds=10)

  j=0
  for traincv, testcv in cv:
    classifier_c = nltk.NaiveBayesClassifier.train(training_sample[traincv[0]:traincv[len(traincv)-1]])
    for k,(attribute,label) in enumerate(training_sample[testcv[0]:testcv[len(testcv)-1]]):
        reference_c.append(label)
        class_predictedc= classifier_c.classify(attribute)
        predicted_c.append(class_predictedc)
    accuracy_c=nltk.classify.util.accuracy(classifier_c, training_sample[testcv[0]:testcv[len(testcv)-1]])
    sum_accuracy+=accuracy_c
    j=j+1
  print(reference_c,predicted_c)
  print ('Confusion Matrix:','\n', nltk.metrics.ConfusionMatrix(reference_c, predicted_c))

  mean_cross_accuracy=  sum_accuracy/j
  print('Cross Validation:',mean_cross_accuracy)
  act_acc = nltk.classify.util.accuracy(classifier, test_sample)
  print ('Accuracy:', act_acc)
  print ('Confusion Matrix:','\n', nltk.metrics.ConfusionMatrix(reference, predicted))

  for j in range(0,5):
      precision =  nltk.metrics.precision(rsets[str(j)], psets[str(j)])
      recall = nltk.metrics.recall(rsets[str(j)], psets[str(j)])
      print ('Precision of class',j,':',precision)
      print ('Recall of class:',j,':',recall)

      if recall is None or precision is None:
          print('No F-1')
      else:
          try:
              f1 = float(2 * ((recall*precision)/(recall+precision)))
          except ZeroDivisionError:
              f1 = 0.0
          print('F1-Measure of class',j,f1)

if __name__ == '__main__':
 training_data = open(sys.argv[1])
 test_data = open(sys.argv[2])

 classify_mnb(training_data,test_data)
