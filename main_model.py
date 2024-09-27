import warnings
warnings.filterwarnings('ignore')
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset,DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class UCDM(nn.Module):
    def __init__(self, num_user, num_item, model_args, device):
        
        super(UCDM, self).__init__()
        
        self.args = model_args
        # init args
        self.embedding_dim = self.args.d
        embedding_dim = self.embedding_dim
        self.L = self.args.L  # sequence length
        L = self.L
        self.w = 0.2  # learnable para
        
        # define embedding
        
        self.user_embed = nn.Embedding(num_user, embedding_dim).to(device)
        self.item_embed = nn.Embedding(num_item, embedding_dim).to(device)
        self.item_embed_2 = nn.Embedding(num_item, embedding_dim).to(device)
        self.item_embed_short = nn.Embedding(num_item, embedding_dim).to(device)
        self.item_embed_long = nn.Embedding(num_item, embedding_dim).to(device)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim).to(device)
        #print(self.linear1)
        self.item_position_embed = nn.Embedding.from_pretrained(self.position_embed(L),freeze=True)
        
         #parameter
        self.b = nn.Embedding(num_item, 1, padding_idx=0).to(device)
        
        # initialize
        self.user_embed.weight.data.normal_(0, 1.0/self.user_embed.embedding_dim)
        self.item_embed.weight.data.normal_(0, 1.0/self.item_embed.embedding_dim)
        self.item_embed_2.weight.data.normal_(0, 1.0/self.item_embed_2.embedding_dim)
        self.linear1.weight.data.normal_(mean=0, std=np.sqrt(2.0 / embedding_dim))
        self.linear2.weight.data.normal_(mean=0, std=np.sqrt(2.0 / embedding_dim))
        
        self.b.weight.data.zero_()
    
    def position_embed(self, L):
        position_embedding = np.array([[pos/np.power(1000, 2.*i)/ self.embedding_dim for i in range(self.embedding_dim)]
                                      for pos in range(L)])
        position_embedding[:,0::2] = np.sin(position_embedding[:,0::2])
        position_embedding[:,1::2] = np.cos(position_embedding[:,1::2])
        t = torch.from_numpy(position_embedding).to(device)
        return t
    
    def att_cluster_block(self, cluster_batch_seq):
        item_embedding_2 = self.item_embed_2(cluster_batch_seq).to(device)
        # item position embedding
        position_idx = torch.range(0, self.L - 1).unsqueeze(0).expand(cluster_batch_seq.size(0), -1).long().to(device)
        position_embedding = self.item_position_embed(position_idx)
        item_embedding_cat_2 = item_embedding_2.float() + position_embedding.float()
        
        # self-attention network
        Q = F.relu(self.linear2(item_embedding_cat_2))
        K = F.relu(self.linear2(item_embedding_cat_2))
        #print(K.shape)
        d = torch.FloatTensor([self.embedding_dim]).to(device)
        affinity = torch.matmul(Q, torch.transpose(K, 1, 2))/torch.sqrt(d)
        
        # mask the diagonal value
        mask = torch.eye(item_embedding_cat_2.size(1), item_embedding_cat_2.size(1)).byte().to(device)
        affinity = affinity.masked_fill(mask.bool(), 0)
        #S = F.softmax(affinity)
        S = torch.sigmoid(affinity)
        #print('s', S.shape)
        attention = torch.mean(torch.matmul(S, item_embedding_cat_2), dim=1)
        #print('user_position is on:', user_position)
        #print(attention[user_position])
        return attention
    
    
    def forward(self, seq_item, user_id, target, user_cluster_seq, sequences_np, user_label, for_pred = False):
        
        '''
        user_id
        seq_item = L item id that user interact before
        target: item target
        '''
        
        # sequential item embedding
        # seq_item = seq_item.to(device)
        item_embedding = self.item_embed(seq_item).to(device)
        # item position embedding
        position_idx = torch.range(0, self.L - 1).unsqueeze(0).expand(seq_item.size(0), -1).long().to(device)
        position_embedding = self.item_position_embed(position_idx)
        item_embedding_cat = item_embedding.float() + position_embedding.float()
        
        # self-attention network
        Q = F.relu(self.linear1(item_embedding_cat))
        K = F.relu(self.linear1(item_embedding_cat))
        #print(K.shape)
        d = torch.FloatTensor([self.embedding_dim]).to(device)
        affinity = torch.matmul(Q, torch.transpose(K, 1, 2))/torch.sqrt(d)
        
        # mask the diagonal value
        mask = torch.eye(item_embedding_cat.size(1), item_embedding_cat.size(1)).byte().to(device)
        affinity = affinity.masked_fill(mask.bool(), 0)
        #S = F.softmax(affinity)
        S = torch.sigmoid(affinity)
        #print('s', S.shape)
        attention = torch.mean(torch.matmul(S, item_embedding_cat), dim=1)
        #print('attention', attention.shape)
        
        #print('im inside!!!!', user_cluster_seq.shape)
        #print('im important', batch_users)
        
        
        # train the cluster_batch
        #attention_cluster = attention
        ### Prepare the user cluster mapping
        #if for_pred == True:
            #print('user_label,', user_label)
        arr_cat = np.array([])
        for i in range(user_cluster_seq.shape[0]):
            arr_cat = np.concatenate((arr_cat, np.array(user_cluster_seq[i])), axis=0)
        # get user_mapping index
        #if for_pred == True:
            #print('ARR', arr_cat)
        mapping_index = np.zeros(len(user_label))
        for i in range(len(user_label)):
            #if for_pred == True:
                #print('---------Test-----Batch----------User', user_id[i].item())
            mapping_index[i] = np.where(arr_cat == user_id[i].item())[0][0]
        #print('get mapping index,', mapping_index)
        # get each batch user's label
    #print('putongniubi buniubi ,user label series', user_label)
        #if for_pred == True:
            #print('---------------Test------------------Index', mapping_index)
        # train the cluster_batch
       
        cluster_batch_seq = torch.from_numpy(sequences_np[user_cluster_seq[0]]).type(torch.LongTensor).to(device)
        cluster_output = self.att_cluster_block(cluster_batch_seq)
        for i in range(1, user_cluster_seq.shape[0]):
            #print('incluster items: ', user_cluster_seq[i])
            cluster_batch_seq = torch.from_numpy(sequences_np[user_cluster_seq[i]]).type(torch.LongTensor).to(device)
            #print('0000000000000000000000', cluster_batch)
            #print(self.att_cluster_block(cluster_batch_seq).size())
            cluster_output = torch.cat((cluster_output, self.att_cluster_block(cluster_batch_seq)), 0)
        cluster_output = cluster_output[mapping_index]
         
        
        # user embedding
        user_id = user_id.to(device)
        user_embedding = self.user_embed(user_id).squeeze()
        b = self.b(target)
        #print('user_embed', user_embedding.shape)
        # target embedding short and long note: those two embedding is different 
        
        if target is None:
            target = torch.range(0,self.num_item-1).long().unsqueeze(0).cuda()
            target_embedding_short = self.item_embed_short(target).squeeze()
            target_embedding_long = self.item_embed_long(target).squeeze()
        else:
            target_embedding_short = self.item_embed_short(target).squeeze()
            target_embedding_long = self.item_embed_long(target).squeeze()
        
        # pred
        if for_pred == False:
            user_embedding = user_embedding.unsqueeze(1).expand(-1,target.size(1),-1)
            #print('to train', user_embedding.shape)
            #print('attention before', attention.shape)
            attention = attention.unsqueeze(1).expand(-1,target.size(1),-1)
            cluster_output = cluster_output.unsqueeze(1).expand(-1,target.size(1),-1)
            
            #print('target_embedding_short', target_embedding_short.shape)
            y_pred = 0.1 * torch.sqrt(torch.sum((user_embedding - target_embedding_long)**2, dim=2)) + 0.8 *torch.sqrt(torch.sum((attention-target_embedding_short)**2, dim=2)) + 0.1 *torch.sqrt(torch.sum((cluster_output-target_embedding_short)**2, dim=2))
            #print('to train----------', y_pred.size())
            return y_pred
        else:
            target = target.unsqueeze(0)
            #print('target_size', target.shape)
            #print('user_embedding', user_embedding.shape)
            user_embedding = user_embedding.unsqueeze(0).expand(target.size(1), -1)
            #print('user_embedding_after', user_embedding.shape)
            #print('attention_before', attention.shape)
            #print('----------------if the final hsere ----------------------------')
            #print('test attention ------------------', attention.size())
            attention = attention.expand(target.size(1), -1)
            #print('test cluster_output ------------------', cluster_output.size())
            cluster_output = cluster_output.expand(target.size(1), -1)
            
            #print('attention', attention.shape)
            y_pred = 0.1 * torch.sqrt(torch.sum((user_embedding - target_embedding_long)**2, dim=1)) + 0.8 *torch.sqrt(torch.sum((attention-target_embedding_short)**2, dim=1)) + 0.1 *torch.sqrt(torch.sum((cluster_output-target_embedding_short)**2, dim=1))
            #print('to test----------', y_pred.size())
            return y_pred, attention, cluster_output