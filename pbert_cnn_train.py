import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import tqdm
import Levenshtein
from scipy.stats import rankdata
from transformers import BertTokenizer, BertModel
import pickle
from torch.utils.data import Dataset, DataLoader
import sys

class ProteinDataset(Dataset):
    def __init__(self, df, pretrained_bert='Rostlab/prot_bert', max_length=512):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        self.max_length = max_length
        self._gen_wt_encodings()
    
    def _gen_wt_encodings(self):
        # make embedding look-up more efficient
        self.wt_enc_dict = {}
        wt_set = set(self.df.sequence)
        for wt in wt_set:
            wt_enc = self.tokenizer(' '.join(wt), padding = 'max_length', max_length = self.max_length,
                                    return_token_type_ids = False, truncation = True, return_tensors = 'pt')
            self.wt_enc_dict[wt] = wt_enc

    def __getitem__(self, item):
        r = self.df.iloc[item]
        mut = r.sequence[: int(r.seq_position)] + r.mutant + r.sequence[int(r.seq_position)+1:]
        mut_enc = self.tokenizer(' '.join(mut), padding = 'max_length', max_length = self.max_length,
                                 return_token_type_ids = False, truncation = True, return_tensors = 'pt')
        features = torch.tensor(r.features, dtype=torch.float32)
        ddG = torch.tensor(r.ddG, dtype=torch.float32)
        dt = torch.tensor(r.dT, dtype=torch.float32)
        mut_mask = torch.zeros(self.max_length)
        if r.seq_position >= self.max_length:
            mut_mask[self.max_length-1] = 1
        else:
            mut_mask[r.seq_position] = 1
        return self.wt_enc_dict[r.sequence], mut_enc, mut_mask, features, ddG, dt

    def __len__(self):
        return len(self.df)

class Model(nn.Module):
    def __init__(self, conv_structure, fc1_structure, custom_bert, conv_device, bert_device, activation='relu'):
        super().__init__()
        self.custom_bert = custom_bert
        self.conv_device = conv_device
        self.bert_device = bert_device
        assert activation in ['relu', 'gelu']
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        self.n_conv = len(conv_structure)
        self.n_fc1 = len(fc1_structure)
        self.dropout = nn.Dropout(p=0.1)
        for i in range(1, len(conv_structure)):
            setattr(self, 'conv_' + str(i), nn.Conv3d(in_channels=conv_structure[i-1],
                                                      out_channels=conv_structure[i],
                                                      kernel_size=(3,3,3)))
        for i in range(1, len(fc1_structure)):
            setattr(self, 'fc1_' + str(i), nn.Linear(fc1_structure[i-1], fc1_structure[i]))
        for i in range(1, len(fc1_structure)):
            setattr(self, 'fc2_' + str(i), nn.Linear(fc1_structure[i-1], fc1_structure[i]))
        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2))

    def device_transfer(self):
        self = self.to(self.conv_device)
        self.custom_bert = self.custom_bert.to(self.bert_device)

    def forward(self, x, wt_enc, mut_enc, mut_mask, att_mask):
        x = x.to(self.conv_device)
        for item in wt_enc:
            wt_enc[item] = wt_enc[item].to(self.bert_device)
        for item in mut_enc:
            mut_enc[item] = mut_enc[item].to(self.bert_device)
        mut_mask = mut_mask.to(self.bert_device)
        bert_feas = self.custom_bert(wt_enc, mut_enc, mut_mask, att_mask)
        bert_feas = bert_feas.to(self.conv_device)
        for i in range(1, self.n_conv-1):
            x = getattr(self, 'conv_' + str(i))(x)
            x = self.act(x)
        x = getattr(self, 'conv_'+str(self.n_conv-1))(x)
        x = self.maxpool(x)
        x = x.flatten().view(bert_feas.shape[0], -1)
        ddG_embed = torch.cat([x, bert_feas], dim=1)
        dt_embed = torch.cat([x, bert_feas], dim=1)

        # generate ddG embed
        for i in range(1, self.n_fc1 - 1):
            ddG_embed = self.dropout(ddG_embed)
            ddG_embed = getattr(self, 'fc1_' + str(i))(ddG_embed)
            ddG_embed = self.act(ddG_embed)
        ddG_embed = getattr(self, 'fc1_' + str(self.n_fc1-1))(ddG_embed)

        # generate dt embed
        for i in range(1, self.n_fc1 - 1):
            dt_embed = self.dropout(dt_embed)
            dt_embed = getattr(self, 'fc2_' + str(i))(dt_embed)
            dt_embed = self.act(dt_embed)
        dt_embed = getattr(self, 'fc2_' + str(self.n_fc1 - 1))(dt_embed)
        
        return ddG_embed, dt_embed

class CustomBert(nn.Module):
    def __init__(self, pretrained_bert='Rostlab/prot_bert', reproject_dim=192, n_layers_to_freeze=22):
        super(CustomBert, self).__init__()
        self.n_layers_to_freeze = n_layers_to_freeze
        self.pretrained_bert = BertModel.from_pretrained(pretrained_bert)
        self.bert_embed_dim = self.pretrained_bert.config.hidden_size
        # self.reproject_wt = nn.Linear(self.bert_embed_dim, reproject_dim)
        # self.reproject_mut = nn.Linear(self.bert_embed_dim, reproject_dim)
        # self.reproject_wt_att = nn.Linear(self.bert_embed_dim, reproject_dim)
        # self.reproject_mut_att = nn.Linear(self.bert_embed_dim, reproject_dim)
        self.reproject = nn.Linear(self.bert_embed_dim * 2, reproject_dim * 3)
        self.freeze_bert_layers()

    def freeze_bert_layers(self):
        for name, param in self.pretrained_bert.named_parameters():
            if not name.startswith('pooler'):
                if name.startswith('encoder'):
                    if int(name.split('.')[2]) <= self.n_layers_to_freeze:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

    def forward(self, wt_enc, mut_enc, mut_mask, att_mask):
        batch_size = wt_enc['input_ids'].shape[0]
        # wt_bert_att_feas = self.pretrained_bert(input_ids=wt_enc['input_ids'].view(batch_size, -1),
        #                                         attention_mask=wt_enc['attention_mask'].view(batch_size,
        #                                                                                      -1)).pooler_output.view(-1,
        #                                                                                                              self.bert_embed_dim)
        # mut_bert_att_feas = self.pretrained_bert(input_ids=mut_enc['input_ids'].view(batch_size, -1),
        #                                          attention_mask=mut_enc['attention_mask'].view(batch_size,
        #                                                                                        -1)).pooler_output.view(
        #     -1, self.bert_embed_dim)
        if att_mask == 'one_hot':
            wt_bert_feas = self.pretrained_bert(input_ids=wt_enc['input_ids'].view(batch_size, -1),
                                                attention_mask=mut_mask.view(batch_size, -1)).pooler_output.view(-1,
                                                                                                                 self.bert_embed_dim)
            mut_bert_feas = self.pretrained_bert(input_ids=mut_enc['input_ids'].view(batch_size, -1),
                                                 attention_mask=mut_mask.view(batch_size, -1)).pooler_output.view(-1,
                                                                                                                  self.bert_embed_dim)
        else:
            wt_bert_feas = self.pretrained_bert(input_ids=wt_enc['input_ids'].view(batch_size, -1),
                                                attention_mask=wt_enc['attention_mask'].view(batch_size, -1)).pooler_output.view(-1,
                                                                                                                                 self.bert_embed_dim)
            mut_bert_feas = self.pretrained_bert(input_ids=mut_enc['input_ids'].view(batch_size, -1),
                                                 attention_mask=mut_enc['attention_mask'].view(batch_size, -1)).pooler_output.view(-1,
                                                                                                                                   self.bert_embed_dim)
        # wt_embed_att = self.reproject_wt_att(wt_bert_att_feas)
        # mut_embed_att = self.reproject_mut_att(mut_bert_att_feas)
        # wt_embed = self.reproject_wt(wt_bert_feas)
        # mut_embed = self.reproject_mut(mut_bert_feas)
        # diff_att = mut_embed_att - wt_embed_att
        # diff = mut_embed - wt_embed
        #
        # return self.reproject_all(torch.cat([wt_embed_att, mut_embed_att, diff_att, wt_embed, mut_embed, diff], dim=1))
        return torch.cat([wt_bert_feas, mut_bert_feas], dim=1)

class TrainConfig:
    batch_size = 8
    model = Model(conv_structure=[14, 16, 32, 64, 80, 96, 128], fc1_structure=[3072, 512, 1],
                  custom_bert=CustomBert(n_layers_to_freeze=22), conv_device='cpu', bert_device='mps')
    # model.load_state_dict(torch.load('conv_part_model_1204.pt'))
    # custom_bert.load_state_dict(torch.load('bert_part_model_1204.pt'))
    # model = Model(conv_structure=[14, 2], fc1_structure=[2304, 1])
    #optimizer = torch.optim.Adam(list(model.parameters()) + list(custom_bert.parameters()), lr=4e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.9,
                                                              mode="min",
                                                              patience=10,
                                                              cooldown=10,
                                                              min_lr=1e-6,
                                                              verbose=True)
    loss_func = nn.MSELoss()
    n_epochs = 10
    print_log_every_n_batch = 1
    dt_loss_pct = 0.01
    save_model_path = 'cnn_bert_model.pt'
    att_mask = 'one_hot'

class Trainer:
    def __init__(self, train_loader, dev_loader, train_config):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.cfg = train_config
        # used for storing models
        self.min_dev_loss = float('inf')
    
    def train_and_eval(self):
        self.cfg.model.device_transfer()
        for i in range(self.cfg.n_epochs):
            print('-------------Epoch {}-------------'.format(i+1))
            self.train_and_eval_for_single_epoch()
            
    def train_and_eval_for_single_epoch(self):
        train_total_loss, train_ddG_loss, train_dt_loss = 0., [], []
        self.cfg.model.train()
        n_batch = 0
        for wt_enc, mut_enc, mut_mask, features, ddG, dt in self.train_loader:

            n_batch += 1
            ddG = ddG.to(self.cfg.model.conv_device)
            dt = dt.to(self.cfg.model.conv_device)
            self.cfg.optimizer.zero_grad()
            ddG_pred, dt_pred = self.cfg.model(features, wt_enc, mut_enc, mut_mask, self.cfg.att_mask)
            ddG_pred = ddG_pred.flatten()
            dt_pred = dt_pred.flatten()
            ddG_loss = self.cfg.loss_func(ddG, ddG_pred)
            train_ddG_loss.append(ddG_loss)
            total_loss = ddG_loss
            dt_loss = self.cfg.loss_func(dt[~torch.isnan(dt)], dt_pred[~torch.isnan(dt)])
            if not dt_loss.isnan():
                total_loss += dt_loss * self.cfg.dt_loss_pct
                train_dt_loss.append(dt_loss)
            total_loss.backward()
            train_total_loss += total_loss
            if n_batch % self.cfg.print_log_every_n_batch == 0:
                #print('Total Loss after {} iterations: {}'.format(n_batch, train_total_loss))
                print('Avg DDG Loss after {} iterations: {}'.format(n_batch, sum(train_ddG_loss) / len(train_ddG_loss)))
                if len(train_dt_loss) > 0:
                    print('Avg Train DT Loss after {} iterations: {}'.format(n_batch, sum(train_dt_loss) / len(train_dt_loss)))
            self.cfg.optimizer.step()
            self.cfg.lr_scheduler.step(sum(train_ddG_loss) / len(train_ddG_loss))
        print()
        print('----------------Train Finished---------------')
        print('Total Train Loss: {}'.format(train_total_loss))
        print('Avg Train DDG Loss: {}'.format(sum(train_ddG_loss) / len(train_ddG_loss)))
        if len(train_dt_loss) > 0:
            print('Avg Train DT Loss: {}'.format(sum(train_dt_loss) / len(train_dt_loss)))

        print('----------------Eval on dev set---------------')
        self.cfg.model.eval()
        dev_total_loss, dev_ddG_loss, dev_dt_loss = 0., [], []
        with torch.no_grad():
            for wt_enc, mut_enc, mut_mask, features, ddG, dt in self.dev_loader:
                ddG_pred, dt_pred = self.cfg.model(features, wt_enc, mut_enc, mut_mask, self.cfg.att_mask)
                ddG = ddG.to(self.cfg.model.conv_device)
                dt = dt.to(self.cfg.model.conv_device)
                ddG_pred = ddG_pred.flatten()
                dt_pred = dt_pred.flatten()
                ddG_loss = self.cfg.loss_func(ddG, ddG_pred)
                dt_loss = self.cfg.loss_func(dt[~torch.isnan(dt)], dt_pred[~torch.isnan(dt)])
                total_loss = ddG_loss
                if not dt_loss.isnan():
                    total_loss += dt_loss * self.cfg.dt_loss_pct
                    dev_dt_loss.append(dt_loss)
                dev_total_loss += total_loss
                dev_ddG_loss.append(ddG_loss)
            print('Total Dev Loss: {}'.format(dev_total_loss))
            print('Avg Dev DDG Loss: {}'.format(sum(dev_ddG_loss) / len(dev_ddG_loss)))
            if len(dev_dt_loss) > 0:
                print('Avg Dev DT Loss: {}'.format(sum(dev_dt_loss) / len(dev_dt_loss)))
            if sum(dev_ddG_loss) / len(dev_ddG_loss) < self.min_dev_loss:
                torch.save(self.cfg.model.state_dict(), self.cfg.save_model_path)
                self.min_dev_loss = sum(dev_ddG_loss) / len(dev_ddG_loss)
                print('Best model saved!')
        print('----------------Eval Finished---------------')
        print()

if __name__ == '__main__':
    # log = open("training.log", "a")
    # sys.stdout = log

    with open('data.pkl', 'rb') as handle:
        df = pickle.load(handle)
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
