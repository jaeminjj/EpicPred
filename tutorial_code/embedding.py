import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import numpy as np
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter,ArgumentTypeError
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
def get_parameter_dict():
    parser = ArgumentParser(description='EpicPred',formatter_class=ArgumentDefaultsHelpFormatter)

    # Data
    parser.add_argument('--count_col', type=str,
                        help='TCR frequency or read count column name')
    parser.add_argument('--model_dir', type=str,
                        help='path or BERT and MLP model')
    parser.add_argument('--sample_data',type=str,
                        help='input data path')
    parser.add_argument('--sample_name',type=str,
                        help='sample file name')
    parser.add_argument('--CDR3',type=str, default='cdr3',
                        help='cdr3 col name')
    parser.add_argument('--output_dir',type=str,
                        help='output')
    parser.add_argument('--epitopes_dir',type=str,
                        help='trained_epitopes path')
    parser.add_argument('--cluster_num',type=int, default=5,help='cluster k')
    p = parser.parse_args()
    p = p.__dict__

    return p


p = get_parameter_dict()
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
import numpy as np
import scipy.spatial.distance as spd
import torch

import libmr


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)
    
    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    #ranked_list = input_score.argsort().ravel()[:-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists(train_class_num,trainloader,device,net):
    scores = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # this must cause error for cifar
            outputs = net(inputs.float())
            for score, t in zip(outputs, targets):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists
class Data(Dataset):
    def __init__(self):
        
        self.x=torch.from_numpy(np.array(test_cls_token_repr.cpu().detach().numpy()))
        self.y=torch.from_numpy(np.array(train_labels))
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    nb_classes = len(categories)
    
    ranked_list = input_score.argsort().ravel()[:-1][:alpha]
    #ranked_list = input_score.argsort().ravel()[:-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    #softmax_prob = softmax(np.array(input_score.ravel()))
    softmax_prob=softmax(input_score.cpu().detach(),dim=1)
    return openmax_prob, softmax_prob


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0, 1, 2, 3 중 하나
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # GPU 점유 비율
session = InteractiveSession(config=config)


p = get_parameter_dict()
print(p)
############# MODEL LOADING #############
############# MODEL LOADING #############
device = "cuda" if torch.cuda.is_available() else "cpu"
mlp_model = torch.load(p['model_dir']+'/v2_max_CLS_SGD_mlp.pt', map_location=device)
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = torch.load(p['model_dir']+'/v2_max_CLS_SGD_bert.pt', map_location=device)

model_name = "BMILab/TCR-BERT-MLM"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoModel
huggingface_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
bert_model = AutoModel.from_pretrained(model_name, token=huggingface_token).to(device)


########## EMBEDDING ################


test=pd.read_csv(p['sample_data'])
test_sequences = [" ".join(list(seq)) for seq in test[p['CDR3']]]
train=pd.read_csv(p['epitopes_dir'])
train=train[train['epitope']!='open-set']
train_sequences = [" ".join(list(seq)) for seq in train['beta']]
label_encoder = LabelEncoder()
test_labels = label_encoder.fit_transform(train['epitope'])
########## EMBEDDING ################

bert_result=[]
batch_size=100
bert_model.eval()
encoded_inputs = tokenizer(test_sequences, truncation=True, padding=True, return_tensors='pt')
input_ids = encoded_inputs['input_ids'].to(device)
attention_mask = encoded_inputs['attention_mask'].to(device)

loss_list=[]
for i in range(0, len(test_sequences), batch_size):
    input_batch = input_ids[i:i+batch_size]
    mask_batch = attention_mask[i:i+batch_size]
    outputs = bert_model(input_batch, attention_mask=mask_batch)
    
    A=outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    for j in A:
        bert_result.append(j)

torch.cuda.empty_cache()
scores = mlp_model(torch.tensor(np.array(bert_result)).to(device))
a=[]
score_a=[]
pred_openmax=[]
for j in scores:
    a.append(softmax(j.cpu().detach().numpy()))
    score_a.append( max(softmax(j.cpu().detach().numpy())))
    pred_openmax.append(np.argmax(softmax(j.cpu().detach().numpy())))
score_df=pd.DataFrame(a)
score_df.index=list(test[p['count_col']])
score_df.columns=label_encoder.inverse_transform(range(0,105))
label_df=np.transpose(pd.DataFrame([score_a,pred_openmax]))
label_df.index=list(test[p['count_col']])
label_df.columns=['score','label']
label_df['label']=list(label_encoder.inverse_transform(pred_openmax))
score_df=pd.DataFrame(a)
score_df.index=list(test[p['count_col']])
score_df.columns=label_encoder.inverse_transform(range(0,105))
np.save(p['output_dir']+'/vector/'+p['sample_name']+'.npy',np.array(bert_result))
score_df.to_csv(p['output_dir']+'/score/'+p['sample_name']+'.csv')

torch.cuda.empty_cache()

kmeans = KMeans(n_clusters=p['cluster_num'], random_state=0)
kmeans.fit(np.array(bert_result))
labels = kmeans.predict(np.array(bert_result))
label_df['label']=list(labels)
label_df.to_csv(p['output_dir']+'/label/'+p['sample_name']+'.csv')
    