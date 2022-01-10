"""
The interface to load log datasets. The datasets currently supported include HDFS data.

Authors:
    LogPAI Team

"""

import pandas as pd
import numpy as np
import re
import sys
from datetime import timezone,timedelta
import os, os.path
from fnmatch import fnmatch
from sklearn.utils import shuffle
from collections import OrderedDict
from datetime import datetime as dt
from dateutil.parser import parse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#from naie.log import logger


def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)

def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential',
    save_csv=False):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        save_csv : True or False

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    #print('====== Input data summary ======')

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()  # ordered dictionary
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['LogContent'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['TemplateId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, train_ratio, split_type)

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            #print('Total: {} instances, train: {} instances, test: {} instances'.format(
            #      x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None)
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    # print('Total: {} instances, {} anomaly, {} normal' \
    #       .format(num_total, num_pos, num_total - num_pos))
    # print('Train: {} instances, {} anomaly, {} normal' \
    #       .format(num_train, num_train_pos, num_train - num_train_pos))
    # print('Test: {} instances, {} anomaly, {} normal\n' \
    #       .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)


def load_NAIE(log_file, label_file=None, save_csv=False):
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    data_dict = OrderedDict()  # ordered dictionary
    for idx, row in struct_log.iterrows():
        try:
            dt_str = row['Date'] + 'T' + row['Time']
        except:
            dt_str = row['DateTime']
        ts = parse(dt_str).timestamp()
        slice_win = ts // 300
        # time_slice = (dt.datetime.fromtimestamp(slice_win * 300) + dt.timedelta(hours=time_zone)).strftime('%Y-%m-%d %H:%M:%S')
        if slice_win not in data_dict:
            data_dict[slice_win] = []
        data_dict[slice_win].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()), columns=['slice_win', 'EventSequence'])

    if save_csv:
        data_df.to_csv('data_instances.csv', index=False)

    if label_file is None:
        # Split training and validation set sequentially
        x_data = data_df['EventSequence'].values
        t_data = list(data_df['slice_win'].values)
        #print('Total: {} instances'.format(x_data.shape[0]))
        return (x_data, t_data)
    else:
        raise NotImplementedError('load_NAIE() only support csv and npz files!')

def to_idx(path):
    log_structured_train = [name for name in os.listdir(path) if fnmatch(name, '*.log_structured.csv')]
    log_structured_train.sort()
    log_dict = OrderedDict()
    # generate the data of 'filename': seqs.
    for file in log_structured_train:
        name = file.split('.')[0]
        (x_train, _) = load_NAIE(os.path.join(path, file))
        if name not in log_dict:
            log_dict[name] = []
        log_dict[name].append(list(x_train))
    # calculate the vocabulary of all the templates
    vocab2idx = {'PAD': 0}
    log_templates_train = [name for name in os.listdir(path) if fnmatch(name, '*.log_templates.csv')]
    log_templates_train.sort()
    for file in log_templates_train:
        template_file = pd.read_csv(os.path.join(path, file), engine='c', na_filter=False, memory_map=True)
        for idx, template_id in enumerate(template_file['EventId'], start=len(vocab2idx)):
            vocab2idx[template_id] = idx
    vocab2idx['UNK'] = len(vocab2idx)
    return vocab2idx, log_dict

def generate_data_for_training(vocab2idx, log_dict_train, window_size=10):
    num_sessions = 0
    inputs = []
    outputs = []
    
    for name, seqs in log_dict_train.items():
        for line in seqs[0]:
            num_sessions += 1
            if len(line) == 1:
                line += line
            if len(line) <= window_size:
                line = line[0:-1] + ['PAD'] * (window_size + 1 - len(line)) + [line[-1]]
            line = tuple([vocab2idx.get(ID, vocab2idx['UNK']) for ID in line]) # template_id --> idx
            for i in range(len(line) - window_size):  # gennerate idx seqs
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])

    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))

    return dataset

def generate_data_for_testing(path_test, vocab2idx, window_size=10):
    num_sessions = 0
    log_structured_test = [name for name in os.listdir(path_test) if fnmatch(name, '*.log_structured.csv')]
    log_dict_test = OrderedDict()
    time_dict_test = OrderedDict()
    # generate the data of 'filename': seqs.
    for file in log_structured_test:
        name = file.split('.')[0]
        (x_train, t_data) = load_NAIE(os.path.join(path_test, file))
        if name not in log_dict_test:
            log_dict_test[name] = []
            # time_dict_test[name] = []
        log_dict_test[name].append(list(x_train))
        time_dict_test[name] = t_data

    for name, seqs in log_dict_test.items():
        dataset = []
        for line in seqs[0]:
            if len(line) == 1:
                line += line
            if len(line) <= window_size:
                line = line[0:-1] + ['PAD'] * (window_size + 1 - len(line)) + [line[-1]]
            line = tuple([vocab2idx.get(ID, vocab2idx['UNK']) for ID in line]) # template_id --> idx
            dataset.append(line)
        log_dict_test[name] = dataset
        num_sessions += len(dataset)
        #print('file:sessions = {}:{}'.format(name, len(dataset)))
    #print('Number of test_files:{}'.format(len(log_dict_test)))
    #print('Number of sessions:{}'.format(num_sessions))
    return log_dict_test, time_dict_test

def result_to_csv(y, t, result_dir, time_zone=8):
    dataset = []
    datetime = []
    label = []
    for name, slices in t.items():
        for i,slice_win in enumerate(slices):
            time_slice = (dt.utcfromtimestamp(slice_win*300) + timedelta(hours=time_zone)).strftime('%Y-%m-%d %H:%M:%S')
            dataset.append(name)
            datetime.append(time_slice)
            label.append(y[name][i])
    data_dict = {'dataset': dataset, 'time_slice(UTC+8)': datetime, 'label':label}
    data_df = pd.DataFrame(data_dict)
    # result_dir = '../result'
    if not os.path.exists(result_dir):                         # 若不存在保存路径，则创建
        os.makedirs(result_dir)
    if os.path.exists(result_dir + 'submit.csv'):        # 若存在历史数据，则清除
        os.remove(result_dir + 'submit.csv')    
    dataset_list = ['E9000刀片服务器下电', 'E9000刀片服务器重启', 'E9000服务器交换板下电',
                    'E9000服务器交换板重启', 'E9000服务器电源板故障', '交换机Eth-trunk端口LACP故障',
                    '交换机端口频繁Up_Down', '存储系统管理链路故障']
    submit_df = pd.DataFrame()
    for name in dataset_list:
        df = data_df[data_df['dataset']==name]
        submit_df = submit_df.append(df, ignore_index=True)
    submit_df.to_csv(result_dir + 'submit.csv', index=False, encoding='utf-8')
    #logger.info('Result was saved to submit.csv')
