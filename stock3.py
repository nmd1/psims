# Get multiple stock data form the market using Yahoo Finance API
import json
import urllib.request
import numpy as np
import pandas as pd

# Valid intervals [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
# Valid ranges [1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
def getMultiFinancialDataJSON(stocks, interval="2m", rge="1d"):
    base = "https://query1.finance.yahoo.com/v8/finance/chart/"
    urls = []
    for stock in stocks:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock}?interval={interval}&range={rge}"
        print("Loading data using URL [",url,"]")
        urls.append(url)
    
    results = []
    for i,url in enumerate(urls):
        try:
            data = urllib.request.urlopen(url).read()
        except urllib.error.HTTPError as e:
            print("Error at stock "+stocks[i]+":",e)
            continue
            
        output = json.loads(data)

        errors = output["chart"]["error"]
        result = output["chart"]["result"]
        
        if(errors):
            msg = f"Error {errors['code']}: {errors['description']}"
            raise RuntimeError(msg)


        #TODO: implies you can add more than one stock to each query, but with no documentation idk how
        result = result[0]
        results.append(result)
    return results 
#s = getMultiFinancialDataJSON("AMD")
#print(s.keys())



def processMultiData(many_data, panda=True,  look_fwd=1):
    if(len(many_data) == 0): return None
    #print("metadata",data["meta"].keys())

    total_index = 0
    total_array = []
    only_one_header_pass = True
    for data in many_data:
        if("close" not in data["indicators"]["quote"][0].keys()):
            print("one of the stocks doesn't have a close parameter")
            continue
        
        prices = data["indicators"]["quote"][0]["close"]
        
        times_list_next = []

        if panda and only_one_header_pass:
            times_list_next.append(np.array(["index", "price"]+["next_price_"+str(i+1) for i in range(look_fwd)]))
            only_one_header_pass = False
            
            
        for i in range(len(prices)-look_fwd):
            array_lst = []
            if panda: array_lst.append(total_index)
            for j in range(look_fwd+1):
                array_lst.append(prices[i+j])
            times_list_next.append(np.array(array_lst))
            total_index+=1

        total_array += times_list_next
    d = np.array(total_array)
    
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
    
def processMultiDataSimple(many_data, panda=True):
    if(len(many_data) == 0): return None
    #print("metadata",data["meta"].keys())

    total_index = 0
    total_array = []
    only_one_header_pass = True
    for data in many_data:
        if("close" not in data["indicators"]["quote"][0].keys()):
            print("one of the stocks doesn't have a close parameter")
            continue
        
        prices = data["indicators"]["quote"][0]["close"]
        
        times_list_next = []

        if panda and only_one_header_pass:
            times_list_next.append(np.array(["index", "price","direction"]))
            only_one_header_pass = False
            
            
        for i in range(len(prices)-1):
            array_lst = []
            if panda: array_lst.append(total_index)
            array_lst.append(prices[i])
            
            
            if(prices[i+1] != None and prices[i] != None):
                delta = prices[i+1]-prices[i]
                if(delta != 0):
                    norm = delta / abs(delta)
                else:
                    norm = 0
                array_lst.append(norm)
            else:
                array_lst.append(None)
            times_list_next.append(np.array(array_lst))
            total_index+=1

        total_array += times_list_next
    d = np.array(total_array)
    
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


def processMultiDataSentiment(many_data, sentiment, panda=True):
    if(len(many_data) == 0): return None
    #print("metadata",data["meta"].keys())

    total_index = 0
    total_array = []
    only_one_header_pass = True
    for stocK_index,data in enumerate(many_data):
        if("close" not in data["indicators"]["quote"][0].keys()):
            print("one of the stocks doesn't have a close parameter")
            continue
        
        prices = data["indicators"]["quote"][0]["close"]
        
        times_list_next = []

        if panda and only_one_header_pass:
            times_list_next.append(np.array(["index","sentiment", "price","direction"]))
            only_one_header_pass = False
            
            
        for i in range(len(prices)-1):
            array_lst = []
            if panda: array_lst.append(total_index)
            array_lst.append(sentiment[stocK_index])
            array_lst.append(prices[i])

            if(prices[i+1] != None and prices[i] != None):
                delta = prices[i+1]-prices[i]
                if(delta != 0):
                    norm = delta / abs(delta)
                else:
                    norm = 0
                array_lst.append(norm)
            else:
                array_lst.append(None)
            times_list_next.append(np.array(array_lst))
            total_index+=1

        total_array += times_list_next
    d = np.array(total_array)
    
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


def processMultiDataSentimentComplex(many_data, sentiment,  panda=True,  look_fwd=1):
    if(len(many_data) == 0): return None
    #print("metadata",data["meta"].keys())

    total_index = 0
    total_array = []
    only_one_header_pass = True
    for stocK_index,data in enumerate(many_data):
        if("close" not in data["indicators"]["quote"][0].keys()):
            print("one of the stocks doesn't have a close parameter")
            continue
        
        prices = data["indicators"]["quote"][0]["close"]
        
        times_list_next = []

        if panda and only_one_header_pass:
            times_list_next.append(np.array(["index","sentiment","price"]+["next_price_"+str(i+1) for i in range(look_fwd)]))
            only_one_header_pass = False
            
            
        for i in range(len(prices)-look_fwd):
            array_lst = []
            if panda: array_lst.append(total_index)
            array_lst.append(sentiment[stocK_index])
            for j in range(look_fwd+1):
                array_lst.append(prices[i+j])
            times_list_next.append(np.array(array_lst))
            total_index+=1

        total_array += times_list_next
    d = np.array(total_array)
    
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
    
    
def getAndProcessMultiFinancialDataJSON(stocks, interval="2m", rge="1d", panda=True, look_fwd=1, simple=True):
    X_total = None
    y_total = None
    first = True
    fd = getMultiFinancialDataJSON(stocks, interval, rge)
    if simple:
        return processMultiDataSimple(fd, panda)
    else:
        return processMultiData(fd, panda, look_fwd)


def getAndProcessMultiFinancialDataJSONSentiment(stocks, sentiment, interval="2m", rge="1d", panda=True, look_fwd=1, simple=True):
    X_total = None
    y_total = None
    first = True
    fd = getMultiFinancialDataJSON(stocks, interval, rge)
    if simple:
        return processMultiDataSentiment(fd,sentiment,panda)
    else:
        return processMultiDataSentimentComplex(fd, sentiment, panda, look_fwd)



#print(getAndProcessMultiFinancialDataJSON(["AMD", "GME", "AMC"], look_fwd=1, panda=True))
#X_raw_2,y_raw_2 = getAndProcessMultiFinancialDataJSONSentiment(["GOOG"],[1],look_fwd=2, interval="2m", rge="1mo")
#print(X_raw_2)
#print(y_raw_2)