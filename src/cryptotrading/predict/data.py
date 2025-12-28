import pandas as pd
from datetime import datetime
from typing import Literal, Optional
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from cryptotrading.predict.utils.timefeatures import time_features
from cryptotrading.data.price import PriceMongoAdapter


class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding window approach"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataFramePriceForecastDataset(Dataset):
    DATE_COLS = ['date', 'datetime', 'timestamp', 'ts', 'time']
    def __init__(self, df, flag='train', size=None,
                 features='S', target='OT', scale=True, 
                 timeenc=0, freq='h'):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            flag (str): 'train', 'test', or 'val'.
            size (list, optional): [seq_len, label_len, pred_len]. Defaults to None.
            features (str, optional): 'S' or 'MS'. Defaults to 'S'.
            target (str, optional): Target feature name. Defaults to 'OT'.
            scale (bool, optional): Whether to scale the data. Defaults to True.
            timeenc (int, optional): Time encoding type. Defaults to 0.
            freq (str, optional): Frequency of the data. Defaults to 'h'.
        """
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.df = df

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.df

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''        
        cols = list(df_raw.columns)
        assert self.target in cols, f"Target {self.target} not in columns {cols}"
        assert any([col in cols for col in self.DATE_COLS]), f"Date not in columns {cols}"
        date_col = [col for col in self.DATE_COLS if col in cols][0]
        cols.remove(self.target)
        cols.remove(date_col)
        df_raw = df_raw[[date_col] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[date_col]][border1:border2]
        df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[date_col].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[date_col].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[date_col].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[date_col].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp[date_col].apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp[date_col].apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop([date_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        elif self.timeenc == 2:
            df_stamp['timestamp'] = df_stamp.date.apply(lambda row: row.timestamp())  # convert to unix timestamp (s)
            data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class CSVPriceForecastDataset(DataFramePriceForecastDataset):
    def __init__(self, csv_path, flag='train', size=None,
                 features='S', target='OT', scale=True, 
                 timeenc=0, freq='h'):
        df = pd.read_csv(csv_path)
        super().__init__(df, flag, size, features, target, scale, timeenc, freq) 

class MongoDBPriceForecastDataset(DataFramePriceForecastDataset):
    def __init__(self, flag: Literal['train', 'test', 'val'] = 'train', 
                 symbol: str = "BTC", 
                 start_time: Optional[datetime] = None, 
                 end_time: Optional[datetime] = None, 
                 limit: Optional[int] = None,
                 size=None, features='S', target='OT', 
                 scale=True, timeenc=0, freq='h'):
        self.db_adapter = PriceMongoAdapter()

        df = self.db_adapter.get_prices(symbol, start_time, end_time, limit)

        super().__init__(df, flag, size, features, target, scale, timeenc, freq) 

def data_provider(args, flag):
    if args.data_path.endswith('.csv'):
        dataset = CSVPriceForecastDataset(args.data_path, flag, args.seq_len, args.label_len, args.pred_len)
    elif args.data_path.endswith('.json'):
        dataset = MongoDBPriceForecastDataset(flag, args.symbol, args.start_time, args.end_time, args.seq_len, args.label_len, args.pred_len)

    return dataset