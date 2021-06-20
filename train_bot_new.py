import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_iris

data = []
#data.append(pd.read_csv("./train_trx2.csv", sep=","))
#data.append(pd.read_csv("./train_btc.csv", sep=","))
#data.append(pd.read_csv("./train_btt.csv", sep=","))
data.append(pd.read_csv("./train_doge.csv", sep=","))
#data.append(pd.read_csv("./train_etc.csv", sep=","))
#data.append(pd.read_csv("./train_eth.csv", sep=","))
#data.append(pd.read_csv("./train_link.csv", sep=","))
#data.append(pd.read_csv("./train_med.csv", sep=","))
data.append(pd.read_csv("./train_vet.csv", sep=","))
#data.append(pd.read_csv("./train_xrp.csv", sep=","))
data.append(pd.read_csv("./train_glm.csv", sep=","))

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
    y = data["current_price"][1:-1]
    X = data.drop(["current_price", "time"], axis=1)[:-2]
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
from sklearn.ensemble import RandomForestRegressor # Bagging + DecisionTree
import pickle

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))

with open('rnd_depth_1.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
#0.9427498088496699
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))
#0.9847028527851172
with open('rnd_depth_2.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))
#0.9939952063199885
with open('rnd_depth_3.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_test, y_test))
#9481924362329752
with open('gbrt_1.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_test, y_test))
#0.9796495474346383
with open('gbrt_2.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_test, y_test))
#0.9980373829084679
with open('gbrt_3.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

from sklearn.ensemble import VotingRegressor


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train, y_train)
print(voting.score(X_test, y_test))
with open('voting_1.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train, y_train)
print(voting.score(X_test, y_test))
with open('voting_2.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train, y_train)
print(voting.score(X_test, y_test))
with open('voting_3.pkl', 'wb') as f:
    pickle.dump(voting, f)


# 0.9940671850507589
# 0.9996090263523396
# 0.9998830598565135
# 0.9513006635495151
# 0.9823924276303277
# 0.9999993611916754
# 0.9856507983808155
# 0.9954605833350809
# 0.9999700375604497
#

# 0.9555544222065252
# 0.9906117192724853
# 0.9974577521812452
# 0.9503588510646396
# 0.9818062070692603
# 0.9999837522992231
# 0.9707193591354013
# 0.9919971529373434
# 0.999343873822353