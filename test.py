import torch
import torch.nn as nn
from models import Transformer
import dataloader
from collections import OrderedDict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

window_size = 10
input_size= 1

def predict(test_loader, model):
    model.eval()
    y_predicted = []
    
    with torch.no_grad():
        for line in test_loader:
            anomaly_flag = 0
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-topN:]
                #possibily_matrix, predicted = torch.sort(output[0])  #升序
                #predicted = predicted[-topN:]
                
                if label not in predicted:
                    anomaly_flag = 1
                    break
            y_predicted.append(anomaly_flag)
    
    return y_predicted

if __name__ == '__main__':
    path_test = 'dataset/structured_logs/test_data'
    path_train = 'dataset/structured_logs/train_data'
    vocab2idx, _ = dataloader.to_idx(path_train)
    vocab_sz = len(vocab2idx)

    model_path = "saved_models/best_model.pth"
    topN = 10

    test_loader, t_loader = dataloader.generate_data_for_testing(path_test, vocab2idx, window_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
            in_dim= input_size,
            embed_dim= 64, 
            out_dim= vocab_sz,
            window_size= window_size,
            depth= 2,
            heads= 8,
            dim_head= 64,
            dim_ratio= 2,
            dropout= 0.1
        )
    
    model = nn.DataParallel(model) # multi-GPU

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    y_predicted = []
    y_hat = OrderedDict()
    
    for name, seqs in test_loader.items():
        y_predicted = predict(seqs, model)
        y_hat[name] = y_predicted
        
    print('Finished Predicting')
    result_dir = 'result/'
    dataloader.result_to_csv(y_hat, t_loader, result_dir, time_zone = 8)
