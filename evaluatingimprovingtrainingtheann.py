import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data=pd.read_csv("Churn_Modelling.csv")
X = data.iloc[: , 3:13].values
Y = data.iloc[: , 13].values

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


X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential 
from keras.layers import Dense
 from keras.layers import Dropout


def build_classifier():
    classifier =Sequential()
    classifier.add( Dense(output_dim=6,init= "uniform",activation="relu",input_dim=11))  
    classifier.add( Dense(output_dim=6,init= "uniform",activation="relu")) 
    classifier.add( Dense(output_dim=1,init= "uniform",activation="sigmoid"))  
    classifier.compile(optimizer="adam",loss= "binary_crossentropy", metrics=["accuracy"]) 
    return classifier
classifier = KerasClassifier(build_fn= build_classifier,batch_size=5,nb_epoch=100)


accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

mean=accuracies.mean()
variance=accuracies.std()


#improving the ANN
#Dropout regularization to reduce overfitting if needed


classifier =Sequential()
 
# adding Input layer and first hidden layer with dropout
classifier.add( Dense(output_dim=6,init= "uniform",activation="relu",input_dim=11))  
classifier.add(Dropout(p=0.1))
# adding second hidden layer
classifier.add( Dense(output_dim=6,init= "uniform",activation="relu")) 
classifier.add(Dropout(p=0.1))

 # adding output layer
 #if  we have more than 1 output layer we use  
classifier.add( Dense(output_dim=1,init= "uniform",activation="sigmoid"))  
 
 #compiling the ann
 #dependent variable has binary outcome we use loss function as binary_crossentropy
 #dependentvariable are more than two outcome then we use loss function as categorical_crossentropy
classifier.compile(optimizer="adam",loss= "binary_crossentropy", metrics=["accuracy"]) 
 

#Tuning the ann
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],'nb_epoch': [100, 500],'optimizer': ['adam', 'rmsprop']} 

grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring="accuracy",cv=10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
