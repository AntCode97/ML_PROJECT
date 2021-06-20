import time

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
import joblib
import datetime
import pyupbit
import csv
from joblib import load



# model = joblib.load(open('rnd_depth_3.pkl','rb'))
# model = joblib.load(open('rnd_depth_3.pkl','rb'))
# model = joblib.load(open('rnd_depth_3.pkl','rb'))
model_name = "gbrt_2"
#model_name="rnd_3"
model = joblib.load(open(f'{model_name}.pkl','rb'))
#f = open(f"./test_{model_name}.csv",'a', newline='')
f = open(f"./trash.csv",'a', newline='')
w = csv.writer(f)

data = pd.read_csv("./test_doge_fall.csv", sep=",")
columns = data.columns
columns= list(columns)
columns.pop(0)
columns.pop(0)

print(columns)
# y = data["current_price"]
# X = data.drop(["current_price", "time"], axis=1)
#X = X.to_dict('list')

coin = "KRW-DOGE"

balance = 10000
wallet = 0
buy_price=0
count = 180
timecount = 0
cnt = True
sum_price = 0
#매도 가격
ask_price = 0
#매수 가격
bid_price = 0
predict_price = 0
current_price =0
sc=load('std_scaler.bin')

def bid(current_price, bid_price) :
    global balance, wallet, cnt,buy_price,w,predict_price
    if current_price == bid_price:
        available_quantity = balance / (current_price * 1.0005)
        wallet += available_quantity
        balance -= available_quantity * 1.0005 * current_price
        cnt = False
        buy_price = round(current_price * 1.0005)
        print("매수 성공!", "구매가: ", buy_price, "구매 갯수: ", wallet, "잔고 :", balance+wallet*current_price )
        w.writerow([timecount,"매수",buy_price, balance, wallet, bid_price,predict_price, balance+wallet*current_price])

def ask(current_price, ask_price):
    global balance, wallet, cnt,buy_price,w, predict_price
    if current_price == ask_price:
        balance += wallet * 0.9995 * current_price
        print("매도 성공! 판매가:", ask_price, "판매 갯수:", wallet, "잔고 :", balance)
        cnt = True
        wallet =0
        w.writerow([timecount,"매도", buy_price, balance, wallet, ask_price,predict_price, balance + wallet * current_price])


#for current_price, orders in zip(y,X):
while timecount<7199:
    #current_price = d['current_price']
    #orders = d
    timecount+=1
#    data.iloc[timecount]

    current_price = np.array(data.iloc[timecount,1])
    orders =[np.array(data.iloc[timecount, 2:])]
    #current_prices = current_price[-1]
    #orderss = orders[-1]
    #print( current_price,orders)

    # 평균 가격을 계산

    if timecount <= count:
        sum_price += current_price
        continue
    else :
        sum_price -= np.array(data.iloc[timecount-1,1])
        sum_price += current_price


    avg = sum_price/count
    orders_scaled = sc.transform(orders)
    predict_price = model.predict(orders_scaled)
    #수수료를 포함해도, 이득을 취할 수 있는 가격일 때 매수
    if round(current_price*1.0005) < round(avg):
        print(current_price, "평균가:",round(avg), avg, predict_price)
        if balance >= round(current_price*1.0005):


            if balance > 5000:
                if cnt:
                    if predict_price[0] <= current_price:
                       bid(current_price, predict_price[0])
                    else:
                        bid(current_price, current_price)
                    print("잔액", balance)

    # 구매 했던 가격보다 가격이 오른 경우 판매해서 이득을 취함
    if current_price > round(buy_price*1.0005) or predict_price[0] > round(buy_price*1.0005):
        #print(current_price ,buy_price)
        if not cnt:

            if predict_price[0] >= current_price:
                ask(current_price, predict_price[0])
            else:
                ask(current_price, current_price)
            print("판매!!!")


    loss = 0.992

    if current_price <   buy_price*loss or predict_price[0] <buy_price*loss:
        if not cnt:
            print("몰락 전 판매!!!", current_price,  buy_price*loss, predict_price[0])
            if predict_price[0] >= current_price:
                ask(current_price, predict_price[0])
            else:
                ask(current_price, current_price)



w.writerow(["총 수익", balance+ wallet*current_price -10000, "잔고",balance+ wallet*current_price ])
print("총 수익", balance+ wallet*current_price -10000, "잔고",balance+ wallet*current_price)