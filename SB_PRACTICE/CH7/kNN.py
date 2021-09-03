# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:58:42 2019

@author: huyquang
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths


print("[INFO] loading images ...")
imagePaths=list(paths.list_images("datasets\\animals"))

sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])

(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.reshape((data.shape[0],3072))

print("[INFO] feature matrix: {:.1f}MB".format(data.nbytes/(1024*1024.0)))

le=LabelEncoder()
labels  =  le.fit_transform(labels)

(trainX,  testX,  trainY,  testY)  =  train_test_split(data,labels,test_size=0.25,random_state=42)

print("[INFO]  evaluating  k-NN  classifier...")

model=  KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
model.fit(trainX,trainY)
model.fit(trainX,  trainY)
print(classification_report(testY,  model.predict(testX),target_names=le.classes_))
