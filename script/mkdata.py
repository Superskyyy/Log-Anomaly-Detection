import os
from tqdm import tqdm
from logparser import Drain

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
    
    # input_dir  = '../dataset/train_data/'
    # output_dir = '../dataset/structured_logs/train_data'  # The output directory of parsing results
    
    # log_file_lst = os.listdir(input_dir)
    # log_file_lst.sort()
    # print(f"train_data: {log_file_lst}")
    # for i in tqdm(range(len(log_file_lst))):
    #     fname, fext = os.path.splitext(log_file_lst[i])
    #     log_format = log_format_dict[fname]
    #     parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
    #     parser.parse(log_file_lst[i])

    # test_data
    
    input_dir  = '../dataset/test_data'
    output_dir = '../dataset/structured_logs/test_data'  # The output directory of parsing results
    
    log_file_lst = os.listdir(input_dir)
    log_file_lst.sort()
    print(f"test_data: {log_file_lst}")
    for i in tqdm(range(len(log_file_lst))):
        fname, fext = os.path.splitext(log_file_lst[i])
        log_format = log_format_dict[fname]
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
        parser.parse(log_file_lst[i])