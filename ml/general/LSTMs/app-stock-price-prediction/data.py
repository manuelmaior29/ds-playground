import torch
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    
    def __init__(self, ticker, attribute, start_date, end_date, lookback):
        self._ticker = ticker
        self._lookback = lookback
        formatted_end_date = end_date.strftime('%Y-%m-%d')
        formatted_start_date = start_date.strftime('%Y-%m-%d')
        self._series = yf.download(ticker, start=formatted_start_date, end=formatted_end_date)[attribute]
        self._data = self._construct_labeled_data()            
    
    def __getitem__(self, idx):
        return torch.tensor(self._data[idx])

    def __len__(self):
        return len(self.data)

    def _construct_labeled_data(self):
        data_raw = self._series
        data = []
        for index in range(len(data_raw) - self._lookback): 
            data.append(data_raw[index: index + self._lookback])
        data = np.array(data)
        x = data[:,:-1,:]
        y = data[:,-1,:]
        return [x, y]