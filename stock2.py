import json
import urllib.request
import numpy as np
import pandas as pd
import csv
import random
def getFinancialDataJSON(stock, interval="2m", rge="1d"):
    base = "https://query1.finance.yahoo.com/v8/finance/chart/"
    #stock = "AMD"
    interval = "2m"
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock}?interval={interval}&range={rge}"
    print("Loading data using URL [",url,"]")
    
    try:
        data = urllib.request.urlopen(url).read()
    except urllib.error.HTTPError as e:
        raise
        
    output = json.loads(data)

    errors = output["chart"]["error"]
    result = output["chart"]["result"]
    
    if(errors):
        msg = f"Error {errors['code']}: {errors['description']}"
        raise RuntimeError(msg)


    #TODO: implies you can add more than one stock to each query, but with no documentation idk how
    result = result[0]
    
    return result 
#s = getFinancialDataJSON("AMD")
#print(s.keys())



def processData(data, panda=True, justTimes=False, look_fwd=1):
    #print("metadata",data["meta"].keys())
    times = data["timestamp"]
    if justTimes: return times[:-1]
    prices = data["indicators"]["quote"][0]["close"]
    
    times_list = []
    times_list_next = []
    for i in range(len(times)):
        times_list.append((times[i], prices[i]))
    
    if panda:
        times_list_next.append(np.array(["index", "price"]+["next_price_"+str(i+1) for i in range(look_fwd)]))
        
        
    for i in range(len(times)-look_fwd):
        array_lst = []
        if panda: array_lst.append(i)
        for j in range(look_fwd+1):
            array_lst.append(prices[i+j])
        times_list_next.append(np.array(array_lst))

    d = np.array(times_list_next)
    # print(d[0])
    # print(d[1])
    # print(d[2])

    #print(d[1:,-1:])
    
    #print(d[1:,1:-1])
    if panda:
        return (pd.DataFrame(data=d[1:,1:-1],
                    index=d[1:,0],
                    columns=d[0,1:-1]),
                pd.DataFrame(data=d[1:,-1:],
                    index=d[1:,0],
                    columns=d[0,-1:])
        )
                
    
    return d 
    


def getAndProcessFinancialDataJSON(stock, interval="2m", rge="1d", panda=True, look_fwd=1, justTimes=False):
    return (processData(getFinancialDataJSON(stock, interval, rge), panda, justTimes,look_fwd))


# print(getAndProcessFinancialDataJSON("AMD", look_fwd=1, panda=True))


def getStockTickers(elements=None):
    stock_ticker_list = []
    count = 0
    with open('stock_tickers.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for i,row in enumerate(spamreader):
            stock_ticker_list.append(row[0])
    
    if(elements):
        return random.sample(stock_ticker_list, elements)

    return stock_ticker_list
    