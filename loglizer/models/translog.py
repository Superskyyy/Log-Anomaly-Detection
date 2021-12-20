import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from loglizer.models.transformer import Transformer
import argparse
import os

#from naie.log import logger
#from naie.context import Context
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
topN = 10


class LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=28):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TransLog(object):
    def __init__(self, vocab_sz, window_size, batch_size, num_epochs):
        super().__init__()
        self.input_size = 1
        self.hidden_size = 64
        self.num_layers = 2
        self.vocab_sz = vocab_sz
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, train_loader, vocab2idx):
        """The training function of DeepLog"""
        #logger.info('Running environment : {}'.format(device))
#        logger.info("Please write your code here to supplement the training function.\n "
#              "Of course, if you don't want to use deeplog, you can also use\n "
#              "other methods, such as './loglizer/models/InvariantsMiner.py', \n"
#              "'./loglizer/models/PCA.py' and so on")
        print('Running environment : {}'.format(device))
        train_loader = DataLoader(train_loader, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        model = Transformer(
            in_dim= self.input_size,
            embed_dim= 64, 
            out_dim= 20,
            window_size= self.window_size,
            depth= 6,
            heads= 8,
            dim_head= 64,
            dim_ratio= 2,
            dropout= 0.1
        ).to(device)
        # seq_dataset = generate_data_for_training('hdfs_train')
        # dataloader = DataLoader(seq_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # Train the model
        start_time = time.time()
        total_step = len(train_loader)
        for epoch in range(self.num_epochs):  # Loop over the dataset multiple times
            train_loss = 0
            for step, (seq, label) in enumerate(train_loader):
                # Forward pass
                seq = seq.clone().detach().view(-1, self.window_size, self.input_size).to(device)
                output = model(seq)
                # print('seq:', seq.shape)
                # print('output:', output.shape)
                # print('label', label.shape)
                loss = criterion(output, label.to(device))
                # print('loss:', loss)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, self.num_epochs, train_loss / total_step))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {:.3f}s'.format(elapsed_time))

        model_path = "../../saved_models"
        print(f'model_path : {model_path}')
        model_name = 'Adam_batch_size{}_epoch{}_{}'.format(str(self.batch_size), str(self.num_epochs), device.type)
        model_dir = 'deeplogmodel'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), model_path + '/' + model_name + '.pt')
        with open(model_path + '/' + model_name + '.ini', "w", encoding='utf-8') as f:
            f.write('[vocab_info]\n')
            for k, v in vocab2idx.items():
                f.write(str(k) + ':' + str(v) + '\n')
        print(f"Finished training, model saved in: {model_path}/{model_name}.pt ")
        return model_path + '/' + model_name + '.pt'

    
    def predict(self, test_loader, model_name):
        """The predict function of DeepLog"""
        preset_model_path = './model/preset_deeplog_model_cuda.pt'
        y_predicted = []
        model = LSTM_Model(self.input_size, self.hidden_size, self.num_layers, self.vocab_sz).to(device)
        #logger.info(f"The sample code will call the preset DeepLog model {preset_model_path}")
        if device.type == 'cpu':
            # GPU (pre-trained model) ==> CPU (Load pre-trained model for predicting)
            model.load_state_dict(torch.load(preset_model_path, map_location=lambda storage, loc: storage))
        else:
            # GPU ==> GPU, CPU ==> CPU
            model.load_state_dict(torch.load(preset_model_path))
        #logger.info(f"You can change the path to the latest training model")
        model.eval()   # set to evaluation mode
        start_time = time.time()
        with torch.no_grad():
            for line in test_loader:
                anomaly_flag = 0
                for i in range(len(line) - self.window_size):
                    seq = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, self.window_size, self.input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-topN:]
                    if label not in predicted:
                        anomaly_flag = 1
                        break
                y_predicted.append(anomaly_flag)
        elapsed_time = time.time() - start_time
        print('elapsed_time: {:.3f}s'.format(elapsed_time))
        print('Finished Predicting')
        return y_predicted
