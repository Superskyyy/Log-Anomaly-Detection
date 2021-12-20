#!/usr/bin/env python
import sys
# sys.path.append('../')
import os, os.path
from fnmatch import fnmatch
from logparser import Drain
from loglizer.models import InvariantsMiner
from loglizer.models import PCA
from loglizer.models import DeepLog
from loglizer import dataloader, preprocessing
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from naie.datasets import get_data_reference
from naie.context import Context
from naie.log import logger
import moxing as mox


if __name__ == '__main__':

    # 1. data collecting
    log_format_dict = {
        'CE交换机五天无故障日志数据集': '<DateTime> <<Pid>><Content>',                                         # train_data
        'E9000服务器五天无故障日志数据集':'<DateTime> <Content>',                                              # train_data
        'OceanStor存储五天无故障日志数据集': '<Invalid_Date> <<Pid>><Date> <Time> <IP> <Component> <Content>', # train_data
        'E9000刀片服务器下电': '<DateTime> <Content>',                                                        # test_data
        'E9000刀片服务器重启': '<DateTime> <Content>',                                                        # test_data
        'E9000服务器电源板故障': '<DateTime> <Content>',                                                      # test_data
        'E9000服务器交换板下电': '<DateTime> <Content>',                                                      # test_data
        'E9000服务器交换板重启': '<DateTime> <Content>',                                                      # test_data
        '存储系统管理链路故障': '<Invalid_Date> <<Pid>><Date> <Time> <IP> <Component> <Content>',              # test_data
        '交换机Eth-trunk端口LACP故障': '<DateTime> <<Pid>><Content>',                                         # test_data
        '交换机端口频繁Up_Down': '<DateTime> <<Pid>><Content>',                                               # test_data
    }

    # 2. template extraction or log parsing
    regex      = [
        r'blk_(|-)[0-9]+' , # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ]     # Regular expression list for optional preprocessing (default: [])
    st    = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes
    # train_data
    data_reference = get_data_reference(dataset="DatasetService", dataset_entity="log_abnormal_training_dataset")
    file_paths = data_reference.get_files_paths()
    
#    input_dir  = './data/raw_logs/NAIE/train_data'
    input_dir  = '/cache/train_data/'
    output_dir = './data/structured_logs/NAIE/train_data'  # The output directory of parsing results
    mox.file.copy_parallel(os.path.dirname(file_paths[0]), input_dir)
    
    log_file_lst = os.listdir(input_dir)
    log_file_lst.sort()
    print(f"train_data: {log_file_lst}")
    for i in tqdm(range(len(log_file_lst))):
        fname, fext = os.path.splitext(log_file_lst[i])
        log_format = log_format_dict[fname]
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
        parser.parse(log_file_lst[i])
    # test_data
    data_reference = get_data_reference(dataset="DatasetService", dataset_entity="log_abnormal_test_dataset")
    file_paths = data_reference.get_files_paths()
    
    input_dir  = '/cache/test_data'
    output_dir = './data/structured_logs/NAIE/test_data'  # The output directory of parsing results
    
    mox.file.copy_parallel(os.path.dirname(file_paths[0]), input_dir)
    
    log_file_lst = os.listdir(input_dir)
    log_file_lst.sort()
    print(f"test_data: {log_file_lst}")
    for i in tqdm(range(len(log_file_lst))):
        fname, fext = os.path.splitext(log_file_lst[i])
        log_format = log_format_dict[fname]
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
        parser.parse(log_file_lst[i])

    # 3. feature extraction and anomaly detection
    # model training
    # train data
    logger.info('structured train data:')
    path_train = './data/structured_logs/NAIE/train_data'
    train_loader, vocab2idx = dataloader.generate_data_for_training(path_train, window_size=10)
    print(f"vocab2idx===>{vocab2idx}")
    vocab_sz = len(vocab2idx)
    model = DeepLog(vocab_sz, window_size=10, batch_size=1024, num_epochs=100)
    model_path = model.fit(train_loader, vocab2idx)
    # test data
    logger.info('structured test data:')
    path_test = './data/structured_logs/NAIE/test_data'
    test_loader, t_loader = dataloader.generate_data_for_testing(path_test, vocab2idx, window_size=10)
    y_hat = OrderedDict()
    for name, seqs in test_loader.items():
        logger.info(f'Testing data : {name}')
        y_predicted = model.predict(seqs, model_path)
        y_hat[name] = y_predicted

    result_dir = './result/DeepLog/'
    dataloader.deeplog_result_to_csv(y_hat, t_loader, result_dir, time_zone = 8)
    mox.file.copy(os.path.join(result_dir, "submit.csv"), os.path.join(Context.get_output_path(), "submit.csv"))

