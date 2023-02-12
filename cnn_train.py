import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import tqdm
import Levenshtein
from scipy.stats import rankdata
import pickle
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    def __init__(self, df=None):
        self.df = df

    def __getitem__(self, item):
        r = self.df.iloc[item]
        return torch.tensor(r.features, dtype=torch.float32), torch.tensor(r.ddG, dtype=torch.float32), torch.tensor(r.dT, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

class Model(nn.Module):
    def __init__(self, conv_structure, fc1_structure, fc2_structure, flatten_size, activation='relu'):
        super().__init__()
        assert activation in ['relu', 'gelu']
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        self.n_conv = len(conv_structure)
        self.n_fc1 = len(fc1_structure)
        self.n_fc2 = len(fc2_structure)
        self.dropout = nn.Dropout(p=0.1)
        for i in range(1, len(conv_structure)):
            setattr(self, 'conv_' + str(i), nn.Conv3d(in_channels=conv_structure[i-1],
                                                        out_channels=conv_structure[i],
                                                        kernel_size=(3,3,3)))
        for i in range(1, len(fc1_structure)):
            setattr(self, 'fc1_' + str(i), nn.Linear(fc1_structure[i-1], fc1_structure[i]))
        for i in range(1, len(fc2_structure)):
            setattr(self, 'fc2_' + str(i), nn.Linear(fc2_structure[i-1], fc2_structure[i]))
        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2))
        self.flatten_size = flatten_size
    
    def forward(self, x):
        for i in range(1, self.n_conv-1):
            x = getattr(self, 'conv_' + str(i))(x)
            x = self.act(x)
        x = getattr(self, 'conv_'+str(self.n_conv-1))(x)
        x = self.maxpool(x)
        dt_embed, ddG_embed = x.flatten().view(-1, self.flatten_size), x.flatten().view(-1, self.flatten_size)
        for i in range(1, self.n_fc1-1):
            dt_embed = self.dropout(dt_embed)
            dt_embed = getattr(self, 'fc1_' + str(i))(dt_embed)
            dt_embed = self.act(dt_embed)
        dt_embed = getattr(self, 'fc1_' + str(self.n_fc1-1))(dt_embed)
        
        for i in range(1, self.n_fc2-1):
            ddG_embed = self.dropout(ddG_embed)
            ddG_embed = getattr(self, 'fc2_' + str(i))(ddG_embed)
            ddG_embed = self.act(ddG_embed)
        ddG_embed = getattr(self, 'fc2_' + str(self.n_fc2-1))(ddG_embed)
        
        return dt_embed, ddG_embed

class TrainConfig:
#     if torch.backends.mps.is_available():
#         device = 'mps'
#     elif torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
    device = 'cpu'
    batch_size = 32
    model = Model(conv_structure=[14, 16, 32, 64, 128, 256, 512], fc1_structure=[4096, 1024, 512, 128, 1],
                  fc2_structure=[4096, 1024, 512, 128, 1], flatten_size=4096)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.9,
                                                              mode="min",
                                                              patience=10,
                                                              cooldown=10,
                                                              min_lr=1e-5,
                                                              verbose=True)
    loss_func = nn.MSELoss()
    n_epochs = 10
    print_log_every_n_batch = 10
    save_model_path = 'cnn_model.pt'
    dt_loss_pct = 0.01


class Trainer:
    def __init__(self, train_loader, dev_loader, train_config):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.cfg = train_config
        # used for storing models
        self.min_dev_loss = float('inf')

    def train_and_eval(self):
        self.cfg.model.to(self.cfg.device)
        for i in range(self.cfg.n_epochs):
            self.train_and_eval_for_single_epoch()

    def train_and_eval_for_single_epoch(self):
        train_total_loss, train_ddG_loss, train_dt_loss = 0., [], []
        self.cfg.model.train()
        n_batch = 0
        for features, ddG, dt in self.train_loader:
            total_loss = torch.tensor(0.)
            n_batch += 1
            self.cfg.optimizer.zero_grad()
            features = features.to(self.cfg.device)
            ddG = ddG.to(self.cfg.device)
            dt = dt.to(self.cfg.device)
            ddG_pred, dt_pred = self.cfg.model(features)
            ddG_pred = ddG_pred.flatten()
            dt_pred = dt_pred.flatten()
            ddG_loss = self.cfg.loss_func(ddG[~torch.isnan(ddG)], ddG_pred[~torch.isnan(ddG)])
            dt_loss = self.cfg.loss_func(dt[~torch.isnan(dt)], dt_pred[~torch.isnan(dt)])
            if not dt_loss.isnan():
                total_loss += dt_loss * self.cfg.dt_loss_pct
                train_dt_loss.append(dt_loss)
            if not ddG_loss.isnan():
                total_loss += ddG_loss
                train_ddG_loss.append(ddG_loss)
            train_total_loss += total_loss
            if n_batch % self.cfg.print_log_every_n_batch == 0:
                # print('Total Loss after {} iterations: {}'.format(n_batch, train_total_loss))
                print('Avg DDG Loss after {} iterations: {}'.format(n_batch, sum(train_ddG_loss) / len(train_ddG_loss)))
                if len(train_dt_loss) != 0:
                    print('Avg DT Loss after {} iterations {}'.format(n_batch, sum(train_dt_loss) / len(train_dt_loss)))
            total_loss.backward()
            self.cfg.optimizer.step()
        print()
        print('----------------Train Finished---------------')
        print('Total Train Loss: {}'.format(train_total_loss))
        print('Avg Train DDG Loss: {}'.format(sum(train_ddG_loss) / len(train_ddG_loss)))
        print('Avg Train DT Loss: {}'.format(sum(train_dt_loss) / len(train_dt_loss)))
        print()
        print('----------------Eval on dev set---------------')
        self.cfg.model.eval()
        dev_total_loss, dev_ddG_loss, dev_dt_loss = 0., [], []
        with torch.no_grad():
            for features, ddG, dt in self.dev_loader:
                features = features.to(self.cfg.device)
                ddG = ddG.to(self.cfg.device)
                dt = dt.to(self.cfg.device)
                ddG_pred, dt_pred = self.cfg.model(features)
                ddG_pred = ddG_pred.flatten()
                dt_pred = dt_pred.flatten()
                total_loss = torch.tensor(0.)
                ddG_loss = self.cfg.loss_func(ddG[~torch.isnan(ddG)], ddG_pred[~torch.isnan(ddG)])
                dt_loss = self.cfg.loss_func(dt[~torch.isnan(dt)], dt_pred[~torch.isnan(dt)])
                if not dt_loss.isnan():
                    total_loss += dt_loss * self.cfg.dt_loss_pct
                    dev_dt_loss.append(dt_loss)
                if not ddG_loss.isnan():
                    total_loss += ddG_loss
                    dev_ddG_loss.append(ddG_loss)
                dev_total_loss += total_loss

            print('Total Dev Loss: {}'.format(dev_total_loss))
            print('Avg Dev DDG Loss: {}'.format(sum(dev_ddG_loss) / len(dev_ddG_loss)))
            if sum(dev_ddG_loss) / len(dev_ddG_loss) < self.min_dev_loss:
                torch.save(self.cfg.model.state_dict(), self.cfg.save_model_path)
                self.min_dev_loss = sum(dev_ddG_loss) / len(dev_ddG_loss)
                print('Best model saved!')
            print('Avg Dev DT Loss: {}'.format(sum(dev_dt_loss) / len(dev_dt_loss)))
        print('----------------Eval Finished---------------')
        print()

if __name__ == '__main__':
    with open('data.pkl', 'rb') as handle:
        df = pickle.load(handle).reset_index()
    df = df.sample(frac=1, random_state=42).reset_index()
    train_pct = 0.9
    sep_idx = int(len(df) * train_pct)
    df_train = df.iloc[:sep_idx]
    df_dev = df.iloc[sep_idx:]
    train_dataset = ProteinDataset(df_train)
    dev_dataset = ProteinDataset(df_dev)
    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=TrainConfig.batch_size, shuffle=False, pin_memory=True)
    trainer = Trainer(train_loader=train_loader, dev_loader=dev_loader, train_config=TrainConfig)
    trainer.train_and_eval()
