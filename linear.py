import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt

from LinearRegressionGradientDescent import LinearRegressionGradientDescent

# Ucitavanje skupa podataka i prikaz prvih 5 redova

pd.set_option('display.max_columns', 17)
pd.set_option('display.width', None)
data = pd.read_csv('data/fuel_consumption.csv')
print(data.head())

# Prikaz konciznih informacija....
print(data.info())
print(data.describe())
print(data.describe(include=[object]))

# Data cleansing
# print(data.loc[data.FUELCONSUMPTION_HWY.isnull()].head())
# fill NaNs with mean value
# data.FUELCONSUMPTION_HWY = data.FUELCONSUMPTION_HWY.fillna(data.FUELCONSUMPTION_HWY.mean())
# print(data.loc[data.FUELCONSUMPTION_HWY.isnull()].head())
# data.TRANSMISSION = data.TRANSMISSION.filna(data.TRANSMISSION.mode()[0]) ovo mora za string samo
# print(data.loc[data.TRANSMISSION.isnull()].head())

# delete NaN rows, and replace ENGINESIZE with mean
data.ENGINESIZE = data.ENGINESIZE.fillna(data.ENGINESIZE.mean())
# data.dropna(inplace=True)
data = data.dropna(subset=['CO2EMISSIONS'])
# data.where(data.FUELCONSUMPTION_HWY.notnull(), inplace=True)
# data.where(data.FUELTYPE.notnull(), inplace=True)
# data.where(data.TRANSMISSION.notnull(), inplace=True)
print(data.info())
print(data['FUELTYPE'].unique())

data.FUELCONSUMPTION_HWY = data.FUELCONSUMPTION_HWY.fillna(data.FUELCONSUMPTION_HWY.mean())
data.FUELCONSUMPTION_COMB = data.FUELCONSUMPTION_COMB.fillna(data.FUELCONSUMPTION_COMB.mean())
data.FUELCONSUMPTION_HWY = data.FUELCONSUMPTION_HWY.fillna(data.FUELCONSUMPTION_HWY.mean())

#Prikaz grafika...
#Zavisnost kontinualnih atributa koriscenjem korelacione matrice
corr_matrix = data.select_dtypes(include=np.number).corr()
sb.heatmap(corr_matrix, annot=True, square=True, fmt='.1f')
# plt.show()

#Zavisnost izlaznog od svih kontinualnih ulaznih, reasejavajuci tacke po Dekartovom koordinatnom

X = data.loc[:, ['MODELYEAR']]
y = data['CO2EMISSIONS']
plt.figure('Dependency CO2EMISSIONS from MODELYEAR')
plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='MODELYEAR')
plt.xlabel('MODELYEAR', fontsize=10)
plt.ylabel('CO2EMISSIONS', fontsize=10)
plt.title('Dependency CO2EMISSIONS from MODELYEAR')
plt.legend()
plt.tight_layout()
# plt.show()

X = data.loc[:, ['ENGINESIZE']]
plt.figure('Dependency v from ENGINESIZE')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='ENGINESIZE')
plt.xlabel('ENGINESIZE', fontsize=10)
plt.ylabel('CO2EMISSIONS', fontsize=10)
plt.title('Dependency CO2EMISSIONS from ENGINESIZE')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['CYLINDERS']]
plt.figure('Dependency CO2EMISSIONS from CYLINDERS')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='CYLINDERS')
plt.xlabel('CYLINDERS', fontsize=10)
plt.ylabel('CO2EMISSIONS', fontsize=10)
plt.title('Dependency CO2EMISSIONS from CYLINDERS')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_CITY']]
plt.figure('Dependency CO2EMISSIONS from FUELCONSUMPTION_CITY')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='FUELCONSUMPTION_CITY')
plt.xlabel('FUELCONSUMPTION_CITY', fontsize=10)
plt.ylabel('CO2EMISSIONS', fontsize=10)
plt.title('Dependency CO2EMISSIONS from FUELCONSUMPTION_CITY')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_HWY']]
plt.figure('Dependency CO2EMISSIONS from FUELCONSUMPTION_HWY')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='FUELCONSUMPTION_HWY')
plt.xlabel('FUELCONSUMPTION_HWY', fontsize=10)
plt.ylabel('CO2EMISSIONS', fontsize=10)
plt.title('Dependency CO2EMISSIONS from FUELCONSUMPTION_HWY')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_COMB']]
plt.figure('Dependency CO2EMISSIONS from FUELCONSUMPTION_COMB')
plt.scatter(X, y, s=23, c='yellow', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='FUELCONSUMPTION_COMB')
plt.xlabel('FUELCONSUMPTION_COMB', fontsize=10)
plt.ylabel('CO2EMISSIONS', fontsize=10)
plt.title('Dependency CO2EMISSIONS from FUELCONSUMPTION_COMB')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_COMB_MPG']]
plt.figure('Dependency CO2EMISSIONS from FUELCONSUMPTION_COMB_MPG')
plt.scatter(X, y, s=23, c='green', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='FUELCONSUMPTION_COMB_MPG')
plt.xlabel('FUELCONSUMPTION_COMB_MPG', fontsize=10)
plt.ylabel('CO2EMISSIONS', fontsize=10)
plt.title('Dependency CO2EMISSIONS from FUELCONSUMPTION_COMB_MPG')
plt.legend()
plt.tight_layout()
# plt.show()

#6. graficki prikaz izlaznog atributa od svakog kategorickog atributa koristeci odgovarajuci tip grafika

plt.figure('Dependency CO2EMISSIONS from MAKE')
sb.barplot(x='MAKE', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)

plt.figure('Dependency CO2EMISSIONS from MODEL')
graph = sb.barplot(x='MODEL', y='CO2EMISSIONS', data=data)
plt.xticks([])

plt.figure('Dependency CO2EMISSIONS from vehicle VEHICLECLASS')
sb.barplot(x='VEHICLECLASS', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)

plt.figure('Dependency co2emission from TRANSMISSION')
sb.barplot(x='TRANSMISSION', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)

plt.figure('Dependency CO2EMISSIONS from FUELTYPE')
sb.barplot(x='FUELTYPE', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)
plt.show()


def absolute_maximum_scale(series):
    return series / series.abs().max()


# feature engineering
# not useful features: MODELYEAR, MAKE, MODEL, TRANSMISSION, VEHICLECLASS?
# useful features: ENGINESIZE, CYLINDERS, , FUELTYPE, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, FUELCONSUMPTION_COMB, FUELCONSUMPTION_COMB_MPG
# normalized = preprocessing.normalize(data.loc[:, ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS' ]])
# print("Normalized: ", normalized)

for col in ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
            'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']:
    data[col] = absolute_maximum_scale(data[col])
data_train = data.loc[:, ['ENGINESIZE', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_HWY',  'CYLINDERS', 'FUELTYPE']]
labels = data.loc[:, 'CO2EMISSIONS']
print(data_train.head())

# ja moram da kodiram vehicleclass i transmission?
ohe = OneHotEncoder(dtype=int, sparse=False)
fueltype = ohe.fit_transform(data_train.FUELTYPE.to_numpy().reshape(-1, 1))
data_train = data_train.drop(columns=['FUELTYPE'])
data_train = data_train.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['FUELTYPE'])))
print(data_train.head())

# print(np.any(np.isnan(data_train))) #and gets False
# print(np.all(np.isfinite(data_train))) #and gets True
# Model training

lr_model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.7, random_state=382, shuffle=False)
lr_model.fit(X_train, y_train)
labels_predicted = lr_model.predict(X_test)
ser_predicted = pd.Series(data=labels_predicted, name="Predicted", index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_predicted], axis=1)

print(res_df)
print('Model score: ', lr_model.score(X_test, y_test))

# Kreiranje i obucavanje modela
lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train, y_train)
print("X_train? ", X_train)
learning_rates = np.array([[0.001], [0.01], [0.01], [0.01], [0.01], [0.0001], [0.0001], [0.001], [0.01]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 100)
# Vizuelizacija modela
plt.figure('MS Error')
plt.plot(np.arange(0, len(mse_history)), mse_history)
plt.title('MS Error')
plt.show()
# plt.show()

#
y_train_copy = y_train.copy(deep=True)
labels_predicted = lrgd.predict(X_test) * (y_train_copy.max() - y_train_copy.min()) + y_train_copy.min()
ser_predicted = pd.Series(data=labels_predicted, name='Predicted', index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_predicted], axis=1)

print("Predikcije linearne regresije metodom gradijentnig spusta:\n", res_df)
# print('score: ', r2_score(np.array(y_test), labels_predicted), end='\n \n')
print('R^2 score: ', r2_score(np.array(y_test), labels_predicted), end='\n \n')
