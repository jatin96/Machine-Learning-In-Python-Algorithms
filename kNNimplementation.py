import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace = True) 
df.drop(df.columns[[0]],1,inplace = True) #delete first column because it's the index . 1 means column and inplace
#means replace the df with the updated one

x = np.array(df.drop(df.columns[[9]],1)) # feature set
y = np.array(df[df.columns[9]]) #class set or label

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2) 

clf = neighbors.KNeighborsClassifier()

clf.fit(x_train,y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

#prediction
example_measure = np.array([[4,2,1,1,1,2,3,2,1]])
prediction = clf.predict(example_measure)
print(prediction)





