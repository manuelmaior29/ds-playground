import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def split_data(stock, lookback):
    data_raw = stock
    data = []
    
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

def fetch_data_last_n_days(ticker, column, n):
    current_date_time = datetime.now()
    start_date_time = current_date_time - timedelta(days=n) 
    formatted_current_date_time = current_date_time.strftime('%Y-%m-%d')
    formatted_start_date_time = start_date_time.strftime('%Y-%m-%d')
    df = yf.download(ticker, start=formatted_start_date_time, end=formatted_current_date_time)
    return df[column].to_numpy()