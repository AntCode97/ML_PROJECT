import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_iris


data = pd.read_csv("./train_trx.csv", sep=",")
#data = np.loadtxt("./trade_list.csv" ,delimiter=',', dtype=np.float)
#print(data)
#print(data[:10])
print(data.shape)
# for i,v in enumerate(data['time']):
#     print(v)
#     hours, minutes = v.split(":")
#     times = hours*3600 + minutes*60
#     data['time'][i] = times

y = data["current_price"][1:]


X = data.drop(["current_price", "time"], axis=1)[:-1]

print("X",X[:10])

print("y" ,y[:10])
print(len(X),len(y))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Bagging + DecisionTree
import pickle

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))

with open('rnd_depth_1.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))

with open('rnd_depth_2.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))

with open('rnd_depth_3.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)


