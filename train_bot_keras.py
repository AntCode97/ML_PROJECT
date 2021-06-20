import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import pathlib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#from keras.layers import Dense
from tensorflow.keras.layers import Dense
print(tf.__version__)


data = []
#data.append(pd.read_csv("./train_trx3.csv", sep=","))
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
    bias = 15
    #y = data["current_price"][1:-1]
    X1 = data.drop(["current_price", "time"], axis=1)[:-10]
    X = []
    count = 0
    while count+bias < len(X1):
        #for i in range(count,bias+count):
        #print(X1.loc[count:count+bias].values)

        a = X1.loc[count:count+bias].values
#        X = pd.concat([X1, a], ignore_index=True)

        #X = pd.concat([X1, a], ignore_index=True, axis=1)
        #print(a)
        X2 = []
        for i in a:
            X2.extend(i)
            #print(i)
        count+=1
        X.append(X2)
    #X = pd.DataFrame(X)
    print(X)
    #X = np.array(X)

    y1 = data["current_price"][bias+1:-9]
    y2 = data["current_price"][bias+2:-8]
    y3 = data["current_price"][bias+3:-7]
    y4 = data["current_price"][bias+4:-6]
    y5 = data["current_price"][bias+5:-5]
    y6 = data["current_price"][bias+6:-4]
    y7 = data["current_price"][bias+7:-3]
    y8 = data["current_price"][bias+8:-2]
    y9 = data["current_price"][bias+9:-1]
    y10 = data["current_price"][bias+10:]
    y = np.column_stack([y1, y2])
    y = np.column_stack([y, y3])
    y = np.column_stack([y, y4])
    y = np.column_stack([y, y5])
    y = np.column_stack([y, y6])
    y = np.column_stack([y, y7])
    y = np.column_stack([y, y8])
    y = np.column_stack([y, y9])
    y = np.column_stack([y, y10])
#    y = pd.DataFrame(y)
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
#    print(X1.dtypes, y1.dtypes)
    X = np.concatenate((X, X1), axis=0)
    y = np.concatenate((y, y1), axis=0)
#    X = pd.concat([X, X1], ignore_index=True)
#    X = np.appand(X, X1)
#    y = np.append(y, y1)
#X = X.dropna(axis=0, how='any')
#y = y.dropna(axis=0, how='any')

print("X",X[:10])
print(len(X[0]))
#print(X[0].shape)

print("y" ,y[:10])
print(len(X),len(y))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Bagging + DecisionTree
import pickle

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from joblib import dump, load
dump(sc, 'sc_scaler.bin', compress=True)


model = keras.Sequential()
model.add(Dense(128, activation='relu', input_shape=[len(X_train[0])]))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='relu'))
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae', 'mse'])
model.summary()

model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
model.summary()

loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
# 1613/1613 - 3s - loss: 0.9864 - mae: 0.7023 - mse: 0.9864
# 테스트 세트의 평균 절대 오차:  0.70 MPG
print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))
result = model.predict(X_test[:10])
print(result)
print(y_test[:10])
#from keras.models import load_model
model.save('my_model.h5')
# rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
# rnd_regressor.fit(X_train, y_train)
# print(rnd_regressor.score(X_test, y_test))
#
# with open('rnd_depth_1.pkl', 'wb') as f:
#     pickle.dump(rnd_regressor, f)
# #0.9427498088496699
# rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
# rnd_regressor.fit(X_train, y_train)
# print(rnd_regressor.score(X_test, y_test))
# #0.9847028527851172
# with open('rnd_depth_2.pkl', 'wb') as f:
#     pickle.dump(rnd_regressor, f)
#
# rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
# rnd_regressor.fit(X_train, y_train)
# print(rnd_regressor.score(X_test, y_test))
# #0.9939952063199885
# with open('rnd_depth_3.pkl', 'wb') as f:
#     pickle.dump(rnd_regressor, f)
#
# rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
# rnd_regressor.fit(X_train, y_train)
# print(rnd_regressor.score(X_test, y_test))
# #0.9939952063199885
# with open('rnd_depth_4.pkl', 'wb') as f:
#     pickle.dump(rnd_regressor, f)



# from sklearn.ensemble import GradientBoostingRegressor
#
# gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
# gbrt.fit(X_train, y_train)
# print(gbrt.score(X_test, y_test))
# #9481924362329752
# with open('gbrt_1.pkl', 'wb') as f:
#     pickle.dump(gbrt, f)
#
# gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
# gbrt.fit(X_train, y_train)
# print(gbrt.score(X_test, y_test))
# #0.9796495474346383
# with open('gbrt_2.pkl', 'wb') as f:
#     pickle.dump(gbrt, f)
#
# gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
# gbrt.fit(X_train, y_train)
# print(gbrt.score(X_test, y_test))
# #0.9980373829084679
# with open('gbrt_3.pkl', 'wb') as f:
#     pickle.dump(gbrt, f)
#
# from sklearn.ensemble import VotingRegressor
#
#
# rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
# gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
# voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
# voting.fit(X_train, y_train)
# print(voting.score(X_test, y_test))
# with open('voting_1.pkl', 'wb') as f:
#     pickle.dump(voting, f)
#
# rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
# gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
# voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
# voting.fit(X_train, y_train)
# print(voting.score(X_test, y_test))
# with open('voting_2.pkl', 'wb') as f:
#     pickle.dump(voting, f)
#
# rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
# gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
# voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
# voting.fit(X_train, y_train)
# print(voting.score(X_test, y_test))
# with open('voting_3.pkl', 'wb') as f:
#     pickle.dump(voting, f)


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