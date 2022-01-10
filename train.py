import os, os.path
from loglizer.models.transformer import Transformer
from loglizer import dataloader, preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    window_size = 10
    batch_size= 1024
    learning_rate= 0.0001
    num_epochs= 300
    input_size= 1

    path_train = 'dataset/structured_logs/train_data'
    vocab2idx, log_dict = dataloader.to_idx(path_train)
    
    vocab_sz = len(vocab2idx)

    data = dataloader.generate_data_for_training(vocab2idx, log_dict, window_size)

    train_size = int(len(data) * 0.8)
    validate_size = len(data) - train_size
    print("Length of dataset: {}".format(len(data)))
    print("Length of training dataset: {}".format(train_size))
    print("Length of validation dataset: {}".format(validate_size))

    train_dataset, val_dataset = random_split(data, [train_size, validate_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size= 1, shuffle= False)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = Transformer(
            in_dim= input_size,
            embed_dim= 64, 
            out_dim= vocab_sz,
            window_size= window_size,
            depth= 6,
            heads= 8,
            dim_head= 64,
            dim_ratio= 2,
            dropout= 0.1
        ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # Train the model
    loss_min = 99999
    model_name = 'best_model.pth'
    model_path = "saved_models"

    save_path = os.path.join(model_path,model_name)
    best_model = model

    print("Begin training ......")

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        val_loss = 0

        # Training
        for step, (seq, label) in enumerate(train_loader):
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        ave_trainloss = train_loss / len(train_loader)

        # Vaildating
        with torch.no_grad():    
            for step, (seq, label) in enumerate(val_loader):
                seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                val_loss += loss.item()
        
        ave_valoss = val_loss / len(val_loader)

        if ave_valoss < loss_min:
            loss_min = ave_valoss
            torch.save(model.state_dict(), save_path)
            best_model = model
            print("Model saved")
    
        print('Epoch [{}/{}], train_loss: {:.14f} val loss: {:.14f}'.format(epoch + 1, num_epochs, ave_trainloss, ave_valoss))

    print(f"Finished training, model saved in: {save_path} ")
