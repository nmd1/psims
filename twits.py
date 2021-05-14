# Not used; grabs stock data, and sentiment, from stocktwits' API 
# Requests are far too rate limited to be useful for this project

import requests
import json
import os.path
from os import path

def getThirtyPosts(ticker, cache=True):
    if(cache and path.exists(f'{ticker}_ticker.json')):
        try:
            json_data = None
            with open(f'{ticker}_ticker.json', 'r') as f:
                    json_data = json.load(f)
            print(json.dumps(json_data, indent=4, sort_keys=True))
            print("Loading JSON from disk")
            return json_data
        except json.decoder.JSONDecodeError as e:
            print("Error parsing file as json!")
    

    
    print("Retreiving JSON from internet")  
     
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    
    # defining a params dict for the parameters to be sent to the API
    #PARAMS = {'address':location}
    
    
    # sending get request and saving the response as response object
    response = requests.get(url=url)# , params=PARAMS)
    json_data = json.loads(response.text)
    print(json.dumps(json_data, indent=4, sort_keys=True))

    #json_data = response.json()
    if(cache):
        print("Saving JSON to disk")  

        with open(f'{ticker}_ticker.json', 'w') as f:
            json.dump(json_data, f)
        

    return json_data

getThirtyPosts("AMD", cache=True)
