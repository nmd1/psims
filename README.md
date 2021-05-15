# psims
## Using Sentiment for Predicting Stocks 

### Set up
Run `pip install -r requirements.txt` to install the needed packages (ideally in a py environment)


### Getting stock predictions
Running `prediction.ipynb` will use sentiment model from tweets to make predictions about the stock market.
In doing so it will train a new LSTM prediction model. 

Running `standalone_prediction.ipynb` will use 10 selected stocks which, as of May 14th 2021, had a consistently 
good or bad month to train on. Their sentiment describes if the stock had a good month or bad month.
In doing so it will train a new LSTM prediction model. 