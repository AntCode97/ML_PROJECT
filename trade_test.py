import time

import joblib
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
xg = xgboost.XGBRegressor()
import joblib
import datetime
import pyupbit
access = "j7lMh0E1rBDj6K0njxvTxeAfv8IMqkvIz4QEVpPw"
secret = "T6xKJkNDdrpOKnLfENlDLwfze398dhy0FHCSTeVx"     # 본인 값으로 변경
upbit = pyupbit.Upbit(access, secret)


def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance']), b["avg_buy_price"]
            else:
                return 0



model = joblib.load(open('xg2.pkl','rb'))


coin = "KRW-TRX"
#print(model.score(X_test, y_test))
balance,_ = get_balance("KRW")
wallet = 100000
buy_price=0
count =3
timecount = 0
cnt = True
print(get_balance(coin))
if get_balance(coin):
    _,buy_price = get_balance(coin)
else :
    buy_price = 0

while True:
    timecount+=1
    df = pyupbit.get_ohlcv(coin, interval="minute1", count=count)
    closes = df['close']
    avg = 0
    for i in closes:
        avg += i
    # 평균 가격을 구함
    avg /= count


    orderbook = pyupbit.get_orderbook(tickers=coin)
    current_price = pyupbit.get_current_price(coin)

    orders = []
    orders.append(orderbook[0]["total_ask_size"])
    orders.append(orderbook[0]["total_bid_size"])
    orderbook_units = orderbook[0]["orderbook_units"]
    # print(orderbook_units)
    for unit in orderbook_units:
        orders.append(unit["ask_price"])
        orders.append(unit["bid_price"])
        orders.append(unit["ask_size"])
        orders.append(unit["bid_size"])

    orders = np.array([orders])
    predict_price =model.predict(orders)
    #print(predict_price)
#    print(orders)

    # if predict_price >= current_price*1.0005:
    #     if balance >= current_price*1.005:
    #         wallet +=1
    #         balance -= current_price*1.0005
    # elif predict_price <= current_price*0.98:
    #     if wallet >=0:
    #         wallet-=1
    #         balance += current_price*0.9995

    if current_price < round(avg*1.0005):
        print(current_price, "평균가:",round(avg), avg)
        if balance >= int(current_price*1.0005):

            #print("총액 : ", balance + wallet * current_price, "|||||", avg, predict_price, current_price)
            balance,_ = get_balance("KRW")
            print("잔액", balance)

            if balance > 5000:

                response = upbit.get_order(coin)
                print(response)
                for r in response:
                    if r["side"] == 'bid':
                        if r['state'] == 'wait':
                            create_time = r['created_at'][:10] + " " + r['created_at'][11:19]
                            dt = datetime.datetime.strptime(create_time, '%Y-%m-%d %H:%M:%S')
                            now = datetime.datetime.now().timestamp()
                            dt = dt.timestamp()
                            if now - dt >= 100:
                                print("주문 취소!!")
                                print(upbit.cancel_order(r['uuid']))
                                cnt = True

                if cnt:
                    print("구매!!!")
                    print(predict_price)
                    print(current_price)
                    wallet += 1

                    if get_balance(coin):
                        _, buy_price = get_balance(coin)
                    buy_price = int(current_price * 1.0005)
                    if round(predict_price[0][0]) < current_price:
                        print("예측 가로 사기")

                        response = upbit.buy_limit_order(coin,  round(predict_price[0][5]) ,20)
                        print(response)
                        if response:
                            cnt = False
                            if get_balance(coin):
                                _, buy_price = get_balance(coin)
                    else :
                        response = upbit.buy_limit_order(coin,  current_price ,20)
                        print(response)
                        if response:
                            cnt = False
                            buy_price = current_price * 1.0005
                            if get_balance(coin):
                                _, buy_price = get_balance(coin)

    if current_price > int(buy_price*1.0005):
        print(current_price ,buy_price)
        if wallet >=1:

            #print("총액 : ", balance + wallet * current_price, "|||||", avg, predict_price, current_price)
            real_coin = get_balance("DOGE")
            if real_coin:
                print("이득 취하기!!!")
                wallet -= 1
                print(upbit.sell_limit_order(coin,current_price ,20))
                #print(upbit.sell_market_order(coin, real_coin))
                balance,_ = get_balance("KRW")
                print("잔액", balance)
                cnt = True
    # if predict_price > current_price:
    #     if wallet >=1:
    #         print("이득 취하기!!!")
    #         print(predict_price)
    #         print(current_price)
    #         time.sleep(1)
    #         current_price = pyupbit.get_current_price(coin)
    #         print(current_price)
    #         if current_price > buy_price*1.05:
    #             wallet -=1
    #             balance += current_price*0.9995
    #             print("총액 : ", balance + wallet * current_price, "|||||", avg, predict_price, current_price)
    #         else :
    #             print("응~~안사~~")
    #             print("총액 : ", balance + wallet * current_price, "|||||", avg, predict_price, current_price)
    loss = 0.996
    loss_cnt = False
    #for p in predict_price[0]:
    for p in predict_price:
        if int(p) <= buy_price*loss:
            loss_cnt =True
            print(upbit.sell_limit_order(coin, current_price, 20))
            print("몰락 시그널 발견!")
            break
    print(current_price, predict_price)
    if current_price <   buy_price*loss or loss_cnt:
        print("몰락한다!!", buy_price, predict_price)
        if wallet >=1:
            print("판매!!!")
            print(predict_price)
            print(current_price)
            wallet-=1
            print("총액 : ", balance + wallet * current_price, "|||||", avg, predict_price, current_price)
            response = upbit.get_order(coin)
            print(response)
            try:
                real_coin,_ = get_balance("DOGE")
                #print(upbit.sell_market_order(coin, real_coin))
            except:
                print(upbit.sell_limit_order(coin,current_price ,20))
            for r in response:
                if r["side"]=='ask':
                    if r['state'] =='wait':
                        print("주문 취소!!")
                        print(upbit.cancel_order(r['uuid']))
            try:
                real_coin,_ = get_balance("DOGE")
                print(upbit.sell_market_order(coin, real_coin))
            except:
                print(upbit.sell_limit_order(coin,current_price ,20))

            print("300초 동안 쉬기!!")
            time.sleep(300)
            buy_price = 0
            cnt =True
            loss_cnt = False

    if timecount==20:
        timecount=0
        print(wallet, balance,"총액 : ", balance + wallet * current_price, avg, "산가격", buy_price)




# order = pd.DataFrame(orders, columns = ["total_ask_size", "total_bid_size",	"ask_price", "bid_price", "ask_size", "bid_size", "ask_price",
#                                         "bid_price", "ask_size", "bid_size", "ask_price", "bid_price",	"ask_size", "bid_size", "ask_price",
#                                         "bid_price","ask_size",	"bid_size",	"ask_price","bid_price", "ask_size", "bid_size", "ask_price",
#                                         "bid_price", "ask_size", "bid_size", "ask_price", "bid_price", "ask_size", "bid_size", "ask_price",
#                                         "bid_price", "ask_size", "bid_size",	"ask_price", "bid_price",	"ask_size", "bid_size", "ask_price",
#                                         "bid_price",	"ask_size",	"bid_size"	,"ask_price", "bid_price",	"ask_size",	"bid_size"	,"ask_price",
#                                         "bid_price",	"ask_size",	"bid_size",	"ask_price",	"bid_price",	"ask_size",	"bid_size",	"ask_price",
#                                         "bid_price",	"ask_size",	"bid_size",	"ask_price",	"bid_price",	"ask_size",	"bid_size"])

#xg.load_model("xg.pkl")
