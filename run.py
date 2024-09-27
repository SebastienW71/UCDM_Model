from main_model import UCDM
from interactions import Interactions
from eval_metrics import *
from negative_sampling import *

import argparse
import logging
import torch
import warnings
warnings.filterwarnings('ignore')
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#check gpu device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#### import the data
dir = 'xxxxxx/'  # insert your file path
checkin_file = 'Brightkite.csv'
df = pd.read_csv(dir + checkin_file)


# POIs encode, and generate encode mapping
poi_cat = pd.Categorical(df['poi_id'])
poi_encode = poi_cat.codes
#generate poi mapping table
poi_mapping = pd.DataFrame({
    'poi_encode': poi_encode,
    'poi_id': df['poi_id']
    })
#drop duplicate
poi_mapping_output = poi_mapping.drop_duplicates()
df['poi_encode'] = poi_encode
df.drop(['poi_id'], axis = 1, inplace = True)

df_input = pd.DataFrame({
    'user_id': df['user_id'],  # user_id offset by 1
    'poi_id': df['poi_encode'],
    #'implicit': np.ones(179468)
})

num_users = df_input['user_id'].nunique()
num_items = df_input['poi_id'].nunique()

from copy import deepcopy 
def convert_data(data):
    df = deepcopy(data)
    data = df.groupby('user_id')['poi_id'].apply(list)
    unique_data = df.groupby('user_id')['poi_id'].nunique()
    return data

seq_data = convert_data(df_input)

#### insert POI Candidate
candidate = pd.read_csv('xxxxxx/bk_candidate.csv')  ### insert your file path
candidate = candidate.merge(poi_mapping_output, on = 'poi_id', how = 'left')
candidate.drop(['poi_id'], axis = 1, inplace = True)
candidate = candidate[candidate['distance'] < 20]
candidate.drop(['Unnamed: 0'], axis = 1, inplace = True)

candidate['filtered_rank'] = candidate.groupby('user_id')['rank'].rank()
max_rank = candidate.groupby('user_id')['filtered_rank'].max()
max_rank = max_rank.reset_index()
candidate = candidate.merge(max_rank, on = 'user_id', how = 'left')
top_ratio = 0.85
candidate = candidate[candidate["filtered_rank_x"] > candidate["filtered_rank_y"]* top_ratio]

# Reduct the candidate
candidate_reduction = pd.DataFrame({
    'user_id': candidate['user_id'],  # user_id offset by 1
    'poi_id': candidate['poi_encode'],
    #'implicit': np.ones(179468)
})
empty_id = []
for i in range(1, num_users):
    if i not in candidate_reduction.user_id.unique():
        empty_id.append(i)
empty_df = pd.DataFrame({
    'user_id': empty_id,
    'poi_id': -1
})
candidate_reduction = pd.concat([empty_df, candidate_reduction], ignore_index=True)

candidate_seq = convert_data(candidate_reduction)
candidate_seq = candidate_seq.tolist()

#### import the user cluster seq
import pickle
def pickle_to_df(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data
def series_to_pickle(series, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(series, f)  
        
user_cluster_seq = pickle_to_df('xxxxx/bk_user_cluster_seq.pkl') #### insert your file path
# import user label
user_label = pickle_to_df('xxxxx/bk_user_label.pkl')  #### insert your file path


### The evaluation function
def evaluation(model, epoch_num, train, test_set, candidate, user_cluster_seq, sequences_np, user_label, topk=20):
    model.eval()
    num_users = train.num_users
    num_items = train.num_items
    batch_size = 1083
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    train_matrix = train.tocsr()
    test_sequences = train.test_sequences.sequences
    epoch_num = str(epoch_num)

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]
        batch_test_sequences = test_sequences[batch_user_index]
        batch_test_sequences = np.atleast_2d(batch_test_sequences)
        batch_test_sequences = torch.from_numpy(batch_test_sequences).type(torch.LongTensor).to(device)
        batch_candidate = candidate[start:end]
        item_ids = torch.from_numpy(item_indexes).type(torch.LongTensor).to(device)

        user_label_sub = user_label[batch_user_index]
        user_label_sub = user_label_sub.values
            #        
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(device) 

        rating_pred = np.empty([batch_user_ids.size(0),item_ids.size(0)])

        for idx in range(batch_user_ids.size(0)):
            uid = torch.tensor([idx])
            reduct = batch_candidate[idx]

            uid_pred, attention, cluster_output = model(batch_test_sequences[idx].unsqueeze(0), uid, item_ids, user_cluster_seq, sequences_np, np.array([user_label_sub[idx]]), True)
            if idx == 1:
                dir = 'emb_output_figure'
                torch.save(attention.sum(dim=1), dir + 'att_u_1_' + epoch_num + '.pkl')
                torch.save(cluster_output.sum(dim=1), dir + 'clt_u_1_' + epoch_num + '.pkl')
            pred = uid_pred.cpu().detach().numpy()

            for i in range(len(reduct)):

                pred[int(reduct[i])] = -100

            rating_pred[idx] = pred

        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0
        
        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        #print(ind)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg

#### Train function

def train_model(model, train_data, test_data, candidate, user_cluster_seq, user_label, config):
    num_users = train_data.num_users
    num_items = train_data.num_items

    # convert to sequences, targets and users
    sequences_np = train_data.sequences.sequences
    targets_np = train_data.sequences.targets
    users_np = train_data.sequences.user_ids
    train_matrix = train_data.tocsr()

    n_train = sequences_np.shape[0]
    logger.info("Total training records:{}".format(n_train))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1
    for epoch_num in range(config.n_iter):

        t1 = time.time()

        # set model to training mode
        model.train()

        np.random.shuffle(record_indexes)

        t_neg_start = time.time()
        negatives_np_multi = generate_negative_samples(train_matrix, config.neg_samples, config.sets_of_neg_samples)
        logger.info("Negative sampling time: {}s".format(time.time() - t_neg_start))

        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_users = users_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            negatives_np = negatives_np_multi[batchID % config.sets_of_neg_samples]
            batch_neg = negatives_np[batch_users]
            
            # batch user cluster label::::
            #print('batch user in train', batch_users)
            user_label_sub = user_label[batch_users]
            user_label_sub = user_label_sub.values
            #
            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)
            batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(device)

            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
            prediction_score = model(batch_sequences, batch_users, items_to_predict, user_cluster_seq, sequences_np, user_label_sub, False)
            #print(prediction_score.shape)
            (targets_prediction, negatives_prediction) = torch.split(
                prediction_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

            # compute the hinge loss
            #loss = torch.mean(F.relu(targets_prediction - negatives_prediction + 0.5), dim=1).unsqueeze(1)
            # BPR
            loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
            # binary cross entropy

            loss = torch.mean(torch.sum(loss))

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # clean the grad, 
            #optimizer.zero_grad()
        epoch_loss /= num_batches

        t2 = time.time()

        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        
        if (epoch_num + 1) % 50 == 0:
            precision, recall, MAP, ndcg = evaluation(model, epoch_num, train_data, test_data, candidate, user_cluster_seq, sequences_np, user_label,  topk=20)
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in MAP))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time.time() - t2))   
    logger.info("\n")
    logger.info("\n")
    #torch.save(model.state_dict(), 'car.pkl')
    #print('model arg save complete')
    
    
# split train test data        
def split_data_sequentially(user_records, test_radio=0.2):
    train_set = []
    test_set = []

    for item_list in user_records:
        len_list = len(item_list)
        num_test_samples = int(math.ceil(len_list * test_radio))
        train_sample = []
        test_sample = []
        for i in range(len_list - num_test_samples, len_list):
            test_sample.append(item_list[i])
            
        for place in item_list:
            if place not in set(test_sample):
                train_sample.append(place)
                
        train_set.append(train_sample)
        test_set.append(test_sample)

    return train_set, test_set
    

def generate_dataset(seq_data, num_users, num_items):
    user_records = seq_data.tolist()
    # split dataset
    train_val_set, test_set = split_data_sequentially(user_records, test_radio=0.2)
    train_set, val_set = split_data_sequentially(train_val_set, test_radio=0.1)

    return train_set, val_set, train_val_set, test_set, num_users, num_items


train_set, val_set, train_val_set, test_set, num_users, num_items = generate_dataset(seq_data, num_users, num_items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)

    # model dependent arguments
    parser.add_argument('--d', type=int, default=100)
    config = parser.parse_args(
        args = [
            '--L', '5',
            '--T', '3',
            '--n_iter', '200',
            '--seed', '1200',
            '--batch_size', '500',
            '--learning_rate', '0.001',
            '--l2', '0.001',
            '--neg_samples', '3',
            '--sets_of_neg_samples', '30'
        ])

    
train = Interactions(train_val_set, num_users, num_items)
train.to_sequence(config.L, config.T)

logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logger.info(config)
model = UCDM(num_users, num_items, config, device).to(device)
train_model(model, train, test_set, candidate_seq, user_cluster_seq, user_label, config)