import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import csv

f = open(f"./train_result.csv",'a', newline='')
w = csv.writer(f)

data = []
#data.append(pd.read_csv("./train_trx2.csv", sep=","))
#data.append(pd.read_csv("./train_btc.csv", sep=","))
#data.append(pd.read_csv("./train_btt.csv", sep=","))
data.append(pd.read_csv("./train_doge.csv", sep=","))
#data.append(pd.read_csv("./train_etc.csv", sep=","))
#data.append(pd.read_csv("./train_eth.csv", sep=","))
#data.append(pd.read_csv("./train_link.csv", sep=","))
#data.append(pd.read_csv("./train_med.csv", sep=","))
#data.append(pd.read_csv("./train_vet.csv", sep=","))
#data.append(pd.read_csv("./train_xrp.csv", sep=","))
#data.append(pd.read_csv("./train_glm.csv", sep=","))


#data = np.loadtxt("./trade_list.csv" ,delimiter=',', dtype=np.float)
#print(data)
#print(data[:10])

# for i,v in enumerate(data['time']):
#     print(v)
#     hours, minutes = v.split(":")
#     times = hours*3600 + minutes*60
#     data['time'][i] = times
def preprocessdata(data):
    data = data.dropna(axis=0, how='any')
    y = data["current_price"][1:]
    X = data.drop(["current_price", "time"], axis=1)[:-1]
    # X = X.astype(float)
    # y = y.astype(float)
    # y = data["current_price"][1:-0]
    # X = data.drop(["current_price", "time"], axis=1)[:-1]
    return X,y

# y = data["current_price"][1:]
# X = data.drop(["current_price", "time"], axis=1)[:-1]
X, y = preprocessdata(data[0])
for i in range(len(data)-1):
    X1, y1 = preprocessdata(data[i+1])
    print(X1.dtypes, y1.dtypes)
    X = pd.concat([X, X1], ignore_index=True)
    y = pd.concat([y, y1], ignore_index=True)

print("X",X[:10])

print("y" ,y[:10])
print(len(X),len(y))


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor # Bagging + DecisionTree
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import pickle

f = open(f"./train_result.csv",'a', newline='')
w = csv.writer(f)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))
y_preds = rnd_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_4", rmse, mae, "none"])
#0.9939952063199885
with open('rnd_4.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train, y_train)
print(voting.score(X_test, y_test))

y_preds = voting.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_4", rmse, mae, "none"])
with open('voting_4.pkl', 'wb') as f:
    pickle.dump(voting, f)


scaler = "sc"
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_4", rmse, mae, f"{scaler}"])

with open(f'rnd_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_4", rmse, mae, f"{scaler}"])
with open(f'voting_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(voting, f)

from sklearn.preprocessing import RobustScaler
scaler = "rb"
sc = RobustScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_4", rmse, mae, f"{scaler}"])

with open(f'rnd_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_4", rmse, mae, f"{scaler}"])
with open(f'voting_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(voting, f)

from sklearn.preprocessing import MinMaxScaler
scaler = "msc"
sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_4", rmse, mae, f"{scaler}"])

with open(f'rnd_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_4", rmse, mae, f"{scaler}"])
with open(f'voting_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(voting, f)

from sklearn.preprocessing import Normalizer
scaler = "nz"
sc = Normalizer()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_4", rmse, mae, f"{scaler}"])

with open(f'rnd_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_4", rmse, mae, f"{scaler}"])
with open(f'voting_{scaler}_4.pkl', 'wb') as f:
    pickle.dump(voting, f)