# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:31:11 2021

@author: Dyotak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("Breast Cancer Prediction.csv")


x=dataset.iloc[:, 1: 10].values
y=dataset.iloc[:, 10].values

#Visualization of data>>>>

plt.figure(figsize=(5,5))

sns.countplot(x='Class', data=dataset, palette='magma')  
datasetNew=dataset.drop(columns=["Sample code number"],axis=1) 

plt.figure(figsize=(20,18))
sns.heatmap(datasetNew.corr(), annot=True,linewidths=.5, cmap="Greens")


datasetNew.groupby('Class').hist(figsize=(10, 10))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

'''from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()'''

'''from sklearn.svm import SVC

classifier=SVC(kernel="linear",random_state=0)
classifier=SVC(kernel="rbf",random_state=0)
classifier=SVC(kernel="poly",random_state=0,degree=9)
classifier=SVC(kernel="sigmoid",random_state=0)'''


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 15, criterion = 'entropy',
                                    random_state = 0)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix =\n",cm)
print("Accuracy of Random Forest Classifier model = ",accuracy_score(y_test, y_pred))

new_patient=classifier.predict(sc.transform([[5,4,5,6,8,4,10,2,3]]))
print("New patient's status: ",new_patient)
if new_patient[0]==4:
    print("TYPE: Malignant")
else:
    print("TYPE: Benign")




