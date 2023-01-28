import pandas as pd
import numpy as np
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
data_train = data.loc[:, ['ENGINESIZE', 'CYLINDERS', 'FUELTYPE', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
                          'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG']]
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

# Ovde moram nesto da podesavam nmp
# fig, axis = plt.subplots(1, 1, figsize=(8, 3), dpi=500)
# tree.plot_tree(decision_tree=dtc_model, max_depth=3, feature_names=data_train.columns, class_names=['CO2EMISSIONS'], fontsize=3, filled=True)
# fig.savefig('tree.png')
# plt.close()
# sb.heatmap(data_train.corr(), annot=True, fmt=.2)
# plt.show()



# Kreiranje i obucavanje modela
lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train, y_train)
learning_rates = np.array([[0.17], [0.0000475], [0.15], [0.2], [0.17], [0.0000475], [0.15], [0.2], [0.17], [0.0000475], [0.15]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 20)
# print(res_coeff)
# Vizuelizacija modela
# Stampanje mse za oba modela
lrgd.set_coefficients(res_coeff)
print(f'LRGD MSE: {lrgd.cost():.2f}')
c = np.concatenate((np.array([lr_model.intercept_]), lr_model.coef_))
lrgd.set_coefficients(c)
print(f'LR MSE: {lrgd.cost():.2f}')
# Restauracija koeficijenata
lrgd.set_coefficients(res_coeff)
# Zapamte se koeficijenti LR modela,
# da bi se postavili LRGD koeficijenti i izracunao LR score.
lr_coef_ = lr_model.coef_
lr_int_ = lr_model.intercept_
lr_model.coef_ = lrgd.coeff.flatten()[1:]
lr_model.intercept_ = lrgd.coeff.flatten()[0]
print(f'LRGD score: {lr_model.score(X_test, y_test):.2f}')
# Restauriraju se koeficijenti LR modela
lr_model.coef_ = lr_coef_
lr_model.intercept_ = lr_int_
print(f'LR score: {lr_model.score(X_test, y_test):.2f}')