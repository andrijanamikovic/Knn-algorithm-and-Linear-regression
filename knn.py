import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sb
import matplotlib.pyplot as plt

from knnImpl import KnnImpl

# Ucitavanje skupa podataka i prikaz prvih 5 redova

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)
data = pd.read_csv('data/cakes.csv')
print(data.head())

# Prikaz konciznih informacija....
print(data.info())
print(data.describe())
print(data.describe(include=[object]))


# Data cleansing
# Nema NaN vrednosti


# Prikaz grafika...



print(data.head())

X = data.loc[:, ['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]
y = data.loc[:, 'type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=382, shuffle=False)

k = np.sqrt(len(data)).astype(int)
if (k % 2) == 0 :
    k = k + 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train) #ubaci u model
prediction = knn.predict(X_test)

print('Preciznost testa: ', knn.score(X_test, y_test))

#My imp;
my_knn = KnnImpl(k, X_train, y_train)
prediction_my = my_knn.predict(X_test)
print('Prenciznost implementacije: ', accuracy_score(y_test, prediction_my))