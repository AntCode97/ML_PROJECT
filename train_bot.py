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
from sklearn.ensemble import RandomForestRegressor # Bagging + DecisionTree
from sklearn.preprocessing import StandardScaler
import pickle

f = open(f"./train_result.csv",'a', newline='')
w = csv.writer(f)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))

y_preds = rnd_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_1", rmse, mae, "none"])

with open('rnd_1.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
#0.9427498088496699
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))
#0.9847028527851172
with open('rnd_2.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
y_preds = rnd_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_2", rmse, mae, "none"])

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
rnd_regressor.fit(X_train, y_train)
print(rnd_regressor.score(X_test, y_test))
y_preds = rnd_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_3", rmse, mae, "none"])
#0.9939952063199885
with open('rnd_3.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_test, y_test))
#9481924362329752
with open('gbrt_1.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

y_preds = gbrt.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_1", rmse, mae, "none"])
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_test, y_test))
y_preds = gbrt.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_2", rmse, mae, "none"])
#0.9796495474346383
with open('gbrt_2.pkl', 'wb') as f:
    pickle.dump(gbrt, f)


gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_test, y_test))

y_preds = gbrt.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_3", rmse, mae, "none"])
#0.9980373829084679
with open('gbrt_3.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

from sklearn.ensemble import VotingRegressor


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train, y_train)
print(voting.score(X_test, y_test))

y_preds = voting.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_1", rmse, mae, "none"])


with open('voting_1.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train, y_train)
print(voting.score(X_test, y_test))

y_preds = voting.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)

w.writerow(["voting_2", rmse, mae, "none"])
with open('voting_2.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train, y_train)
print(voting.score(X_test, y_test))

y_preds = voting.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_3", rmse, mae, "none"])
with open('voting_3.pkl', 'wb') as f:
    pickle.dump(voting, f)


model = keras.Sequential()
model.add(Dense(62, activation='relu', input_shape=[len(X_train.columns)]))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
optimizer = tf.keras.optimizers.Adam(lr=0.1)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.fit(X_train, y_train, epochs=30, batch_size=20, verbose=1)
model.summary()
loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
# 1613/1613 - 3s - loss: 0.9864 - mae: 0.7023 - mse: 0.9864
# 테스트 세트의 평균 절대 오차:  0.70 MPG


y_preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))
print("RMSE", rmse)
w.writerow(["keras", rmse, mae, "none"])
#from keras.models import load_model
model.save('my_model.h5')


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from joblib import dump, load
dump(sc, 'sc_scaler.bin', compress=True)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))

y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_1", rmse, mae, "sc"])

with open('rnd_sc_1.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
#0.9427498088496699
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
#0.9847028527851172
with open('rnd_sc_2.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_2", rmse, mae, "sc"])

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_3", rmse, mae, "sc"])
#0.9939952063199885
with open('rnd_sc_3.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
#9481924362329752
with open('gbrt_sc_1.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)

w.writerow(["gbrt_1", rmse, mae, "sc"])


gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)

#0.9796495474346383
with open('gbrt_sc_2.pkl', 'wb') as f:
    pickle.dump(gbrt, f)
w.writerow(["gbrt_2", rmse, mae, "sc"])

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)

#0.9980373829084679
with open('gbrt_sc_3.pkl', 'wb') as f:
    pickle.dump(gbrt, f)
w.writerow(["gbrt_3", rmse, mae, "sc"])
from sklearn.ensemble import VotingRegressor


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)

w.writerow(["voting_1", rmse, mae, "sc"])

with open('voting_sc_1.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)


with open('voting_sc_2.pkl', 'wb') as f:
    pickle.dump(voting, f)
w.writerow(["voting_2", rmse, mae, "sc"])

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_3", rmse, mae, "sc"])
with open('voting_sc_3.pkl', 'wb') as f:
    pickle.dump(voting, f)

model = keras.Sequential()
model.add(Dense(62, activation='relu', input_shape=[len(X_train.columns)]))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
optimizer = tf.keras.optimizers.Adam(lr=0.1)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.fit(X_train_scaled, y_train, epochs=30, batch_size=20, verbose=1)
model.summary()

loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=2)
# 1613/1613 - 3s - loss: 0.9864 - mae: 0.7023 - mse: 0.9864
# 테스트 세트의 평균 절대 오차:  0.70 MPG
print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))
y_preds = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
w.writerow(["keras", rmse, mae, "sc"])
#from keras.models import load_model
model.save('my_model_sc.h5')

from sklearn.preprocessing import RobustScaler
rc = RobustScaler()
X_train_scaled = rc.fit_transform(X_train)
X_test_scaled = rc.transform(X_test)
dump(rc, 'rb_scaler.bin', compress=True)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))

y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_1", rmse, mae, "rb"])
with open('rnd_rb_1.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
#0.9427498088496699
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
#0.9847028527851172
with open('rnd_rb_2.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_2", rmse, mae, "rb"])

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_3", rmse, mae, "rb"])
#0.9939952063199885
with open('rnd_rb_3.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
#9481924362329752
with open('gbrt_rb_1.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_1", rmse, mae, "rb"])

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_2", rmse, mae, "rb"])
#0.9796495474346383
with open('gbrt_rb_2.pkl', 'wb') as f:
    pickle.dump(gbrt, f)


gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_3", rmse, mae, "rb"])
#0.9980373829084679
with open('gbrt_rb_3.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

from sklearn.ensemble import VotingRegressor


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)

w.writerow(["voting_1", rmse, mae, "rb"])

with open('voting_rb_1.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)

w.writerow(["voting_2", rmse, mae, "rb"])
with open('voting_rb_2.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_3", rmse, mae, "rb"])
with open('voting_rb_3.pkl', 'wb') as f:
    pickle.dump(voting, f)


model = keras.Sequential()
model.add(Dense(62, activation='relu', input_shape=[len(X_train.columns)]))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
optimizer = tf.keras.optimizers.Adam(lr=0.1)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.fit(X_train_scaled, y_train, epochs=30, batch_size=20, verbose=1)
model.summary()

loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=2)
# 1613/1613 - 3s - loss: 0.9864 - mae: 0.7023 - mse: 0.9864
# 테스트 세트의 평균 절대 오차:  0.70 MPG
print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))
y_preds = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
w.writerow(["keras", rmse, mae, "rb"])
#from keras.models import load_model
model.save('my_model_rb.h5')

from sklearn.preprocessing import MinMaxScaler

msc = MinMaxScaler()
X_train_scaled = msc.fit_transform(X_train)
X_test_scaled = msc.transform(X_test)
from joblib import dump, load
dump(msc, 'msc_scaler.bin', compress=True)


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))

y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_1", rmse, mae, "msc"])
with open('rnd_msc_1.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
#0.9427498088496699
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
#0.9847028527851172
with open('rnd_msc_2.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_2", rmse, mae, "msc"])

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["rnd_3", rmse, mae, "msc"])
#0.9939952063199885
with open('rnd_msc_3.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
#9481924362329752
with open('gbrt_msc_1.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_1", rmse, mae, "msc"])

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_2", rmse, mae, "msc"])
#0.9796495474346383
with open('gbrt_msc_2.pkl', 'wb') as f:
    pickle.dump(gbrt, f)


gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["gbrt_3", rmse, mae, "msc"])
#0.9980373829084679
with open('gbrt_msc_3.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

from sklearn.ensemble import VotingRegressor


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_1", rmse, mae, "msc"])


with open('voting_msc_1.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_2", rmse, mae, "msc"])

with open('voting_msc_2.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("RMSE: ", rmse)
print("MAE: ", mae)
w.writerow(["voting_3", rmse, mae, "msc"])
with open('voting_msc_3.pkl', 'wb') as f:
    pickle.dump(voting, f)


model = keras.Sequential()
model.add(Dense(62, activation='relu', input_shape=[len(X_train.columns)]))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
optimizer = tf.keras.optimizers.Adam(lr=0.1)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.fit(X_train_scaled, y_train, epochs=30, batch_size=20, verbose=1)
model.summary()

loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=2)
# 1613/1613 - 3s - loss: 0.9864 - mae: 0.7023 - mse: 0.9864
# 테스트 세트의 평균 절대 오차:  0.70 MPG
print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))
y_preds = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
w.writerow(["keras", rmse, mae, "msc"])
#from keras.models import load_model
model.save('my_model_msc.h5')

from sklearn.preprocessing import Normalizer
nz = Normalizer()
X_train_scaled = nz.fit_transform(X_train)
X_test_scaled = nz.transform(X_test)
dump(nz, 'nz_scaler.bin', compress=True)


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))

y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["rnd_1", rmse, mae, "nz"])
with open('rnd_nz_1.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
#0.9427498088496699
rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
#0.9847028527851172
with open('rnd_nz_2.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["rnd_2", rmse, mae, "nz"])

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
rnd_regressor.fit(X_train_scaled, y_train)
print(rnd_regressor.score(X_test_scaled, y_test))
y_preds = rnd_regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["rnd_3", rmse, mae, "nz"])
#0.9939952063199885
with open('rnd_nz_3.pkl', 'wb') as f:
    pickle.dump(rnd_regressor, f)

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
#9481924362329752
with open('gbrt_nz_1.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["gbrt_1", rmse, mae, "nz"])

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))
y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["gbrt_2", rmse, mae, "nz"])

#0.9796495474346383
with open('gbrt_nz_2.pkl', 'wb') as f:
    pickle.dump(gbrt, f)


gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
gbrt.fit(X_train_scaled, y_train)
print(gbrt.score(X_test_scaled, y_test))

y_preds = gbrt.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["gbrt_3", rmse, mae, "nz"])

#0.9980373829084679
with open('gbrt_nz_3.pkl', 'wb') as f:
    pickle.dump(gbrt, f)

from sklearn.ensemble import VotingRegressor


rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=1)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.015)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["voting_1", rmse, mae, "nz"])

with open('voting_nz_1.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.02)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["voting_2", rmse, mae, "nz"])


with open('voting_nz_2.pkl', 'wb') as f:
    pickle.dump(voting, f)

rnd_regressor = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
voting = VotingRegressor(estimators=[('rf', rnd_regressor),('gb', gbrt)])
voting.fit(X_train_scaled, y_train)
print(voting.score(X_test_scaled, y_test))

y_preds = voting.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
print("nz_RMSE: ", rmse)
print("nz_MAE: ", mae)
w.writerow(["voting_3", rmse, mae, "nz"])

with open('voting_nz_3.pkl', 'wb') as f:
    pickle.dump(voting, f)


model = keras.Sequential()
model.add(Dense(62, activation='relu', input_shape=[len(X_train.columns)]))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
optimizer = tf.keras.optimizers.Adam(lr=0.1)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.fit(X_train_scaled, y_train, epochs=30, batch_size=20, verbose=1)
model.summary()

loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=2)
# 1613/1613 - 3s - loss: 0.9864 - mae: 0.7023 - mse: 0.9864
# 테스트 세트의 평균 절대 오차:  0.70 MPG

y_preds = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)
w.writerow(["keras", rmse, mae, "nz"])

model.save('my_model_nz.h5')

