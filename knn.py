import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
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

#4. graficki prikaz zavisnosti kontinualnih atributa koriscenjem korelacione matrice
ohe = OneHotEncoder(dtype=int, sparse=False)
fueltype = ohe.fit_transform(data.type.to_numpy().reshape(-1, 1))
data = data.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['type'])))

corr_matrix = data.select_dtypes(include=np.number).corr()
sb.heatmap(corr_matrix, annot=True, square=True, fmt='.1f')
plt.show()

#5. graficki prikaz zavisnosti izlaznog atributa od svakog ulaznog atributa rasejavajuci tacke
X = data.loc[:, ['flour']]
y = data['type']
sb.catplot(data=data, x="type", y="flour", kind="box")

X = data.loc[:, ['eggs']]
y = data['type']
sb.catplot(data=data, x="type", y="eggs", kind="box")

X = data.loc[:, ['sugar']]
y = data['type']
sb.catplot(data=data, x="type", y="sugar", kind="box")

X = data.loc[:, ['milk']]
y = data['type']
sb.catplot(data=data, x="type", y="milk", kind="box")

X = data.loc[:, ['butter']]
y = data['type']
sb.catplot(data=data, x="type", y="butter", kind="box")

X = data.loc[:, ['baking_powder']]
y = data['type']
sb.catplot(data=data, x="type", y="baking_powder", kind="box")
plt.show()

#6. graficki prikaz zavisnosti izlaznog atributa od svakog ulaznog kategorickog atributa koristeci odg tip grafika


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
y_test1 = y_test.copy(deep = True)
for i in range(0, len(y_test)):
    if (y_test.iloc[i] == "muffin"):
        y_test1.iloc[i] = 1
    else:
        y_test1.iloc[i] = 0
for i in range(0, len(prediction)):
    if (prediction[i] == "muffin"):
        prediction[i] = 1
    else:
        prediction[i] = 0

print ('Greska test: ', mean_squared_error(y_test1, prediction))

#My imp;
my_knn = KnnImpl(k, X_train, y_train)
prediction_my = my_knn.predict(X_test)

print('Prenciznost implementacije: ', accuracy_score(y_test, prediction_my))
for i in range(0, len(prediction_my)):
    if (prediction_my[i] == "muffin"):
        prediction_my[i] = 1
    else:
        prediction_my[i] = 0
print ('Greska implementacija: ', mean_squared_error(y_test1, prediction_my))