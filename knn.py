# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 02:32:39 2019

@author: Philip Arturo Cesani
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

#Manipulating Train
train.head()
train.shape

labels = train['label'].values

unique_val = np.array(labels)
np.unique(unique_val)

train.drop('label', axis = 1, inplace = True)

images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

#Split Training Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Creating KNN from training data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

#Getting Accuracy
from sklearn.metrics import accuracy_score
print("Training accuracy is: ")
print(accuracy_score(y_test,y_pred))

#Manipulating Test Data
test.head()
test.shape

labels = test['label'].values

unique_val = np.array(labels)
np.unique(unique_val)

test.drop('label', axis = 1, inplace = True)

images = test.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

#Split Testing
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier.fit(x_train, y_train)

#Predicting Values from the testing set
y_pred = classifier.predict(x_test)

#Checking Accuracy
from sklearn.metrics import accuracy_score
print("Testing accuracy is: ")
print(accuracy_score(y_test,y_pred))
