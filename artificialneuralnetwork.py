# Artificial neural network
#install Thenao
#install Tensorflow
#install Keras

#importing libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the data set

data=pd.read_csv("Churn_Modelling.csv")
X = data.iloc[: , 3:13].values
Y = data.iloc[: , 13].values

#encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])


labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# removing dummy variable trap

X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras library and packages

import keras
from keras.models import Sequential 
from keras.layers import Dense

#intializing the ANN
classifier =Sequential()
 
# adding Input layer and first hidden layer
classifier.add( Dense(output_dim=6,init= "uniform",activation="relu",input_dim=11))  

# adding second hidden layer
classifier.add( Dense(output_dim=6,init= "uniform",activation="relu")) 
 

 # adding output layer
 #if  we have more than 1 output layer we use  
classifier.add( Dense(output_dim=1,init= "uniform",activation="sigmoid"))  
 
 #compiling the ann
 #dependent variable has binary outcome we use loss function as binary_crossentropy
 #dependentvariable are more than two outcome then we use loss function as categorical_crossentropy
classifier.compile(optimizer="adam",loss= "binary_crossentropy", metrics=["accuracy"]) 
 
 
 #fitting the ann to the training set
classifier.fit(X_train,y_train,batch_size=5,nb_epoch=100 )
 
 #prediction the test result
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
 #making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# predicting a single new observation

"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

cm = confusion_matrix(y_test, y_pred)
