import pyupbit
import numpy as np
import datetime
import pandas as pd
import sys
from pyupbit.request_api import _call_public_api

# access =
# secret =    # 본인 값으로 변경
upbit = pyupbit.Upbit(access, secret)

def get_current_price(ticker="KRW-BTC"):
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
                price = content['trade_price']
                timestamp = content['timestamp']
                ret[market] = price
                ret["timestamp"] = timestamp
            return ret
        else:
            ret = {}
            market = contents[0]['market']
            price = contents[0]['trade_price']
            timestamp = contents[0]['timestamp']
            ret[market] = price
            ret["timestamp"] = timestamp
            return ret
    except Exception as x:
        print(x.__class__.__name__)

from collections import defaultdict
import time
from datetime import datetime
orderbook_dict = defaultdict(list)
current_price_dict = defaultdict(list)
trades_dict = defaultdict(list)
# ask는 매도 bid는 매수
count = 0
orderbook_list = []
current_price_list = []
import csv
f = open("./train_trx2.csv",'a', newline='')
w = csv.writer(f)
coin = "KRW-TRX"
while count< 720000000:
    print(count)
    orderbook = pyupbit.get_orderbook(tickers=coin)
    current_price = get_current_price(coin)
    #print(orderbook[0]['timestamp'])
    #orderbook_list.append(orderbook)
    # if orderbook[0]['timestamp'] ==current_price['timestamp']:
    #     print(orderbook)
    #     print(current_price)
    # d = datetime.fromtimestamp(orderbook[0]['timestamp'])
    # d2 =datetime.fromtimestamp(current_price['timestamp'])
    # print(d, d2)
    # orderbook_dict[d].append(orderbook)
    # current_price_dict[d2].append(current_price)

    orders = []
    orders.append(datetime.now())
    orders.append(current_price[coin])
    orders.append(orderbook[0]["total_ask_size"])
    orders.append(orderbook[0]["total_bid_size"])
    orderbook_units = orderbook[0]["orderbook_units"]
    #print(orderbook_units)
    for unit in orderbook_units:
        orders.append(unit["ask_price"])
        orders.append(unit["bid_price"])
        orders.append(unit["ask_size"])
        orders.append(unit["bid_size"])

    #trades_dict[datetime.now()].append(orders)
    #trades_dict[datetime.now()].append(current_price)
    #print(orders)
    print(orders)
    time.sleep(1)
    count+=1
    w.writerow(orders)

#f.close()
print(get_current_price(coin))
# print(orderbook_dict)
# print(current_price_dict)
#
# print(orderbook_dict.keys())
# print(current_price_dict.keys())

