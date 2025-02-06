import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')




class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='Close', scale=True, timeenc=0, freq='h', scaler = None, external_scaler = False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
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
        self.scaler = StandardScaler()

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.external_scaler= external_scaler
        if self.external_scaler == True:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

    
        self.external_scaler = external_scaler
        self.__read_data__()

    def __read_data__(self):
        # self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # num_train = int(len(df_raw) * 0.7)
        # num_test = int(len(df_raw) * 0.2)
        # num_vali = len(df_raw) - num_train - num_test

        # border1s = [len(df_raw) - num_train, len(df_raw) - num_vali - num_train, 0]
        # border2s = [len(df_raw), len(df_raw) - num_train, num_test]



        num_train = int(len(df_raw) * 0.7)
        print(len(df_raw))
        num_test = int(len(df_raw) * 0.2)

        num_vali = len(df_raw) - num_train - num_test

        border1s = [len(df_raw) - num_train, len(df_raw) - num_vali - num_train, 0]
        border2s = [len(df_raw), len(df_raw) - num_train, num_test]
        print(border1s)
        print(border2s)


        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # if self.scale:
        #     print('scaling @dataloader.py')
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        
        self.last_date = df_raw[['date']].iloc[-self.label_len]

        if self.scale:
            if self.external_scaler == True:
                data = self.scaler.transform(df_data.values)
            else:
                self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            print('not scaling @dataloader.py')

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values


        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


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
    
    def get_last(self):
        s_begin = len(self.data_x) - self.seq_len
        s_end = len(self.data_x)

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        if r_end >= len(self.data_y):
            # print('get_last @lata_loader.py')
            seq_y_zeros = np.zeros_like(seq_x)
            seq_y_label = seq_x[-self.label_len:]
            seq_y = np.concatenate((seq_y_label, seq_y_zeros), axis=0)

            last_label_item = self.last_date

            future_dates = pd.date_range(start=last_label_item.iloc[0], periods=self.pred_len + self.label_len, freq='5T')
            df_future_dates = pd.DataFrame(future_dates, columns=['date'])
            df_stamp = df_future_dates



            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop('date').values

            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, data_stamp




    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_scaler(self):
        return self.scaler
    
    def transform(self, data):
        return self.scaler.transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None , scaler = None, external_scaler = False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.external_scaler= external_scaler
        if self.external_scaler == True:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

    
        self.external_scaler = external_scaler
        self.__read_data__()



    def __read_data__(self):
        


        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        
        
        self.last_date = df_raw[['date']].iloc[-self.label_len]



        if self.scale:
            if self.external_scaler == True:
                data = self.scaler.transform(df_data.values)
            else:
                self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            # seq_y = self.data_x[r_begin:r_begin + self.label_len]
            seq_y = self.data_y[r_begin:r_end]

        else:
            # seq_y = self.data_y[r_begin:r_begin + self.label_len]
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    




    def get_last(self):
        s_begin = len(self.data_x) - self.seq_len
        s_end = len(self.data_x)

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        if r_end >= len(self.data_y):
            # print('get_last @lata_loader.py')
            seq_y_zeros = np.zeros_like(seq_x)
            seq_y_label = seq_x[-self.label_len:]
            seq_y = np.concatenate((seq_y_label, seq_y_zeros), axis=0)

            last_label_item = self.last_date

            future_dates = pd.date_range(start=last_label_item.iloc[0], periods=self.pred_len + self.label_len, freq='5T')
            df_future_dates = pd.DataFrame(future_dates, columns=['date'])
            df_stamp = df_future_dates



            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop('date').values

            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, data_stamp



    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_scaler(self):
        return self.scaler
    
    def transform(self, data):
        return self.scaler.transform(data)
    

    # s_begin--------------18---------------rbegin-------------------18--------------s_end------------------------------36----------------------r_end
    # ----------------------------------------------------------------------------------