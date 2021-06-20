import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import joblib
import datetime
import pyupbit
import csv
from pyupbit.request_api import _call_public_api
from pyupbit.errors import UpbitError, TooManyRequests, raise_error
# access =
# secret =     # 본인 값으로 변경
upbit = pyupbit.Upbit(access, secret)

#CHANGE: EVEN, RISE, FALL
def get_change(ticker="KRW-BTC"):
    """
    최종 체결 가격 조회 (현재가)
    :param ticker:
    :return:
    """
    try:
        url = "https://api.upbit.com/v1/ticker"
        contents = _call_public_api(url, markets=ticker)[0]
        if not contents:
            return None

        if isinstance(ticker, list):
            ret = {}
            for content in contents:
                market = content['market']
                change = content['change']
                ret[market] = change
            return ret
        else:
            return contents[0]['change']
    except Exception as x:
        print(x.__class__.__name__)

def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance']), b["avg_buy_price"]
            else:
                return 0




# model = joblib.load(open('rnd_depth_3.pkl','rb'))
# model = joblib.load(open('rnd_depth_3.pkl','rb'))
# model = joblib.load(open('rnd_depth_3.pkl','rb'))
model = joblib.load(open('gbrt_3.pkl','rb'))
f = open(f"./trash.csv",'a', newline='')
w = csv.writer(f)


coin = "KRW-DOGE"

balance =0
wallet = 0
buy_price=0
count = 10
timecount = 0
cnt = True
sum_price = 0
#매도 가격
ask_price = 0
#매수 가격
bid_price = 0

available_quantity=0
current_price =0

#time 시간동안 거래되지 않은 거래를 취소
def cancel_order(order, time, current_price): #order 는 bid or ask
    global  w
    response = upbit.get_order(coin)
    print(response)
    for r in response:
        if r["side"] == order:
            if r['state'] == 'wait':
                if r["price"] != current_price:
                    create_time = r['created_at'][:10] + " " + r['created_at'][11:19]
                    dt = datetime.datetime.strptime(create_time, '%Y-%m-%d %H:%M:%S')
                    now = datetime.datetime.now().timestamp()
                    dt = dt.timestamp()
                    print("주문 후 경과 시간", now - dt)
                    if now - dt >= time:
                        print("주문 취소!!")
                        w.writerow(["주문 취소", dt, now,now -dt, time, r['side'], r['price']])
                        print(upbit.cancel_order(r['uuid']))

def bid(current_price, bid_price) :
    global balance, wallet, cnt,buy_price
    if current_price == bid_price:
        available_quantity = balance / (current_price * 1.0005)
        wallet += available_quantity
        balance -= available_quantity * 1.0005 * current_price
        cnt = False
        buy_price = current_price * 1.0005
        print("매수 성공!", "구매가: ", buy_price, "구매 갯수: ", wallet, "잔고 :", balance+wallet*current_price )

def ask(current_price, ask_price):
    global balance, wallet, cnt,buy_price
    if current_price == ask_price:
        print("매도 성공!", "구매가: ", buy_price, wallet, "잔고 :", balance+0.9995*current_price, "이득: ", balance + wallet*buy_price)
        balance += wallet * 0.9995 * current_price
        cnt = True
        wallet =0

response = []
sum_prices = []
wait_uuid = 0
#for current_price, orders in zip(y,X):
while True:
    timecount += 1
    df = pyupbit.get_ohlcv(coin, interval="minute1", count=count)
    closes = df['close']
    avg = 0
    for i in closes:
        avg += i
    # 평균 가격을 구함
    avg /= count
    orderbook = pyupbit.get_orderbook(tickers=coin)
    current_price = pyupbit.get_current_price(coin)
    change = get_change(coin)
    print(change)

    #
    # if timecount <= count:
    #     sum_prices.append(current_price)
    #     try:
    #         sum_price += current_price
    #     except:
    #         count+=1
    #
    #     continue
    # else :
    #     sum_price -= sum_prices[0]
    #     sum_prices.pop(0)
    #     sum_price += current_price
    #     sum_prices.append(current_price)
    # #평균 가격을 구함
    # avg = sum_price/count


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
    predict_price = model.predict(orders)

    balance = upbit.get_balance("KRW")

    #수수료를 포함해도, 이득을 취할 수 있는 가격일 때 매수
    if round(current_price*1.0005) < round(avg):
        print(current_price, "평균가:",round(avg), predict_price)
        if balance >= round(current_price*1.0005):

            if cnt:
                cancel_order("bid", 1000, current_price)
                if balance > 5000:

                    if get_balance(coin):
                        _, buy_price = get_balance(coin)



                    if predict_price[0] <= current_price:
                        available_quantity = int(balance / (predict_price[0] * 1.0005))-1
                        response = upbit.buy_limit_order(coin, round(predict_price[0]), available_quantity)
                        print("예측가 매수 시도",response)
                        buy_price = predict_price[0]
                        # if response:
                        #     # cnt = False
                        #     if get_balance(coin):
                        #         _, buy_price = get_balance(coin)
                    else:
                        available_quantity = int(balance / (current_price * 1.0005))-1
                        response = upbit.buy_limit_order(coin,  current_price ,available_quantity)
                        #response = upbit.buy_market_order(coin, balance-100)
                        buy_price = current_price
                        print("현재가 매수 시도",response, current_price)
                        # if response:
                        #     # cnt = False
                        #     buy_price = current_price * 1.0005
                        #     if get_balance(coin):
                        #         _, buy_price = get_balance(coin)
                    print("잔액", balance)

    # 구매 했던 가격보다 가격이 오른 경우 판매해서 이득을 취함
    if round(current_price*0.9995) > buy_price or round(predict_price[0]*0.9995) > buy_price:
        #print(current_price ,buy_price)
        if not cnt:
            real_coin = get_balance(coin[4:])
            if real_coin:
                available_quantity = real_coin[0]
                cancel_order("ask", 3000, current_price)
                if predict_price[0] >= current_price:
                    print("이득 취하기!!!")
                    print(upbit.sell_limit_order(coin, round(predict_price[0]), available_quantity))
                    # print(upbit.sell_market_order(coin, real_coin))
                    balance, _ = get_balance("KRW")
                    print("잔액", balance)
                    # cnt = True
                else:
                    print("이득 취하기!!!")
                    print(real_coin)
                    print(upbit.sell_limit_order(coin, current_price, available_quantity))
                    #print(upbit.sell_market_order(coin, real_coin))
                    # print(upbit.sell_market_order(coin, real_coin))
                    balance, _ = get_balance("KRW")
                    print("잔액", balance)
                    # cnt = True
                print("판매!!!")


    loss = 0.992

    if current_price <   buy_price*loss or predict_price[0] <buy_price*loss:
        if not cnt:
            print("몰락 전 판매!!!", current_price, predict_price, buy_price)
            # if predict_price[0] >= current_price:
            #     response = upbit.sell_limit_order(coin, round(predict_price[0]), available_quantity)
            #     print(response)
            #     # if response:
            #     #     cnt = True
            # else:
            cancel_order("ask", 200, current_price)
            try:
                real_coin, _ = get_balance(coin[4:])
                print(upbit.sell_market_order(coin, real_coin))
            except:
                print(upbit.sell_limit_order(coin, current_price, available_quantity))
            #response = upbit.sell_limit_order(coin, current_price, available_quantity)
            #print(response)

            if current_price < buy_price*0.95:
                response = upbit.get_order(coin)
                for r in response:
                    if r["side"] == 'ask':
                        if r['state'] == 'wait':
                            print(upbit.cancel_order(r['uuid']))
                try:
                    real_coin, _ = get_balance(coin[4:])
                    print(upbit.sell_market_order(coin, real_coin))
                except:
                    print(upbit.sell_limit_order(coin, current_price, available_quantity))
                # if response:
                #     cnt = True
            # response = upbit.get_order(coin)
            # print(response)
            # try:
            #     real_coin, _ = get_balance(coin[4:])
            #     # print(upbit.sell_market_order(coin, real_coin))
            # except:
            #     print(upbit.sell_limit_order(coin, current_price, available_quantity))
            # for r in response:
            #     if r["side"] == 'ask':
            #         if r['state'] == 'wait':
            #             print("주문 취소!!")
            #             print(upbit.cancel_order(r['uuid']))
            # try:
            #     real_coin, _ = get_balance(coin[4:])
            #     print(upbit.sell_market_order(coin, real_coin))
            # except:
            #     print(upbit.sell_limit_order(coin, current_price, available_quantity))
    response = upbit.get_order(coin, state="done")
    print("완료된 거래", response)
    print(cnt, current_price, predict_price, avg, balance, "구매가:", buy_price)
    wait = upbit.get_order(coin, state="wait")
    print("대기 중인 거래", wait)
    if wait:
        if wait_uuid != wait[0]['uuid']:
            wait_uuid = wait[0]['uuid']
            w.writerow(wait[0])
    if response:
        r = response[0]
        if r["side"] == 'bid':
            cnt = False
        else :
            cnt = True
            if r['ord_type'] !='market':
                buy_price = float(r['price'])


    time.sleep(0.9)

#print(""balance+ wallet*current_price)