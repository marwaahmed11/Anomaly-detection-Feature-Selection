import time
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data.csv")
x=data.iloc[:, 2:32 ]
#y=data['diagnosis']

y2=data['diagnosis']=='M'
X_train, X_test, y_train, y_test = train_test_split(x, y2, test_size=0.33)


def wrapper(x,y):
   
    sbs = SFS(LinearRegression(),
         k_features=15, 
         forward=False, 
         floating=False,
         scoring='neg_mean_squared_error',
         cv=0) 
    sbs.fit(x, y)
    return sbs.k_feature_names_  
   


def run(X_train, X_test, y_train, y_test):    
    clf= KNeighborsClassifier(n_neighbors=5)
    clf1= DecisionTreeClassifier()
    clf2= svm.SVC()

    clf.fit(X_train, y_train)
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred1=clf1.predict(X_test)
    y_pred2=clf2.predict(X_test)

    print("Using KNeighborsClassifier: ")
    print("Accuracy on test set: ",accuracy_score(y_test, y_pred))
    print("Precision : ",precision_score(y_test, y_pred ,average=None))
    print("Recall : ",recall_score(y_test, y_pred ,average=None))

    print("Using Decision tree classifier: ")
    print("Accuracy on test set: ",accuracy_score(y_test, y_pred1))
    print("Precision : ",precision_score(y_test, y_pred1 ,average=None))
    print("Recall : ",recall_score(y_test, y_pred1 ,average=None))

    print("Using SVM classifier: ")
    print("Accuracy on test set: ",accuracy_score(y_test, y_pred2))
    print("Precision : ",precision_score(y_test, y_pred2 ,average=None))
    print("Recall : ",recall_score(y_test, y_pred2 ,average=None))


    
 
constant_filter=VarianceThreshold(threshold=6)
constant_filter.fit(X_train)


X_train_filter =constant_filter.transform(X_train)
X_test_filter =constant_filter.transform(X_test)

print(X_train_filter.shape)

print('Using Filter technique: ')
print('New Features: ')
run(X_train_filter,X_test_filter,y_train,y_test)

print('Old Features: ')
run(X_train,X_test,y_train,y_test)

print('----------------------------------------------------------------------------')

X_train_wrapper=X_train.loc[:,wrapper(x,y2)]
X_test_wrapper=X_test.loc[:,wrapper(x,y2)]

print('Using wrapper technique: ')
print('New Features: ')
run(X_train_wrapper,X_test_wrapper,y_train,y_test)

print('Old Features: ')
run(X_train,X_test,y_train,y_test)
