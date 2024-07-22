import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import os
from sklearn.metrics import f1_score
import itertools
import torch.nn.functional as F
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.preprocessing import LabelEncoder
import pickle
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter,ArgumentTypeError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.metrics import roc_auc_score
import copy
import torch.nn.functional as F
from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support

def get_parameter_dict():
    parser = ArgumentParser(description='EpicPred',formatter_class=ArgumentDefaultsHelpFormatter)

    # Data
    parser.add_argument('--metadata_dir', type=str,
                        help='metadata directory')
    parser.add_argument('--output_dir', type=str,default='None',
                        help='output directory first')
    parser.add_argument('--output_dir1', type=str,default='None',
                        help='output directory second')
    parser.add_argument('--input_dir', type=str,default='None',
                        help='input directory')
    parser.add_argument('--sample_info_dir', type=str,default='None',
                        help='sample_info_dir')
    parser.add_argument('--freq', type=str,default='frequency',
                        help='weight of abundance column name in each dataset')
    parser.add_argument('--frequency_or_not',type=int,default=0,
                        help='input data path')
    parser.add_argument('--threshold',type=float,
                        help='threshold of filtering non-binding TCRs')
    parser.add_argument('--cluster_num',type=int, default=5,help='cluster k')
    p = parser.parse_args()
    p = p.__dict__

    return p
p = get_parameter_dict()
metadata_dir=p['metadata_dir']
frequency_or_not=p['frequency_or_not']
freq=p['freq']
threshold=p['threshold']
output_dir=p['output_dir']
output_dir1=p['output_dir1']
cluster_num=p['cluster_num']
sample_info_dir=p['sample_info_dir']
save_inner=0 ## save attention score or not
save_outer=0 ## save attention score or not
#elements=['Severe','Moderate']
length=100
length_or_not=0 ## cutting vectors with length or not
vector_name='vector' # vector_raw
label_name='label' # label2 , label_raw
score_name='score' #score2 ## clustering, score_raw ## raw_clustering
dataset_name='tutorial'
save_outer_name='outer_attention'
save_inner_name='inner_attention'
save_score_name='label' #score2 ## clustering, score_raw ## raw_clustering
lab='WHO_label'
elements=['Severe','Healthy']
sample_name_col='sample'
alpha = 0.001
beta = 1
epochs=1000
input_dir=p['input_dir']

label_encoder = LabelEncoder()


dataset_name='tutorial'
def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)
    
    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total
    class_report_df['avg / total'] = avg

    return class_report_df.T


class EpicPred(nn.Module):
    def __init__(self, cluster_num, class_num, num_heads=4, n_class=2):
        super(EpicPred, self).__init__()
        
        self.embedding_net = nn.Sequential(
            nn.Linear(768, 64),  # Example dimension from BERT output
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        
        self.intra_cluster_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)
        )
        
        self.inter_cluster_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)
        )
        
        self.multihead_attention_1 = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, dropout=0.2)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, dropout=0.2)

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1)
        )
        
        # 각 인스턴스에 대한 예측을 위한 분류기
        self.patch_classifier = nn.Sequential(
            nn.Linear(64, n_class),
            nn.Softmax(dim=1)
        )
        
        self.cluster_num = cluster_num

    def forward(self, x, label, method, count_value, barcode):
        cluster_features = []
        cluster_weights = []
        attention_weight = []
        attention_barcode = []
        instance_preds = []  # 각 인스턴스의 예측 저장
        
        for i in range(self.cluster_num):
            cluster_data = x[label == i]
            cluster_barcode = barcode[label == i]
            if cluster_data.size(0) > 0:
                cluster_embedding = self.embedding_net(cluster_data)
                attn_output, attn_output_weights = self.multihead_attention_1(cluster_embedding, cluster_embedding, cluster_embedding)
                inter_attention_scores = self.intra_cluster_attention(attn_output)
                inter_attention_weights = torch.softmax(inter_attention_scores, dim=0)
                final_feature_representation = (attn_output * inter_attention_weights).sum(dim=0, keepdim=True)
                    
                cluster_features.append(final_feature_representation)
                attention_weight.append(inter_attention_weights)
                attention_barcode.append(cluster_barcode)

                # 각 인스턴스의 예측 수행
                for instance_embedding in attn_output:
                    instance_pred = self.patch_classifier(instance_embedding.unsqueeze(0))
                    instance_preds.append(instance_pred)
                
        all_cluster_features = torch.cat(cluster_features, dim=0)
        
        attn_output, attn_output_weights = self.multihead_attention(all_cluster_features, all_cluster_features, all_cluster_features)
        
        inter_attention_scores = self.inter_cluster_attention(attn_output)
        inter_attention_weights = torch.softmax(inter_attention_scores, dim=0)
        final_feature_representation = (attn_output * inter_attention_weights).sum(dim=0, keepdim=True)

        y_pred = self.fc6(final_feature_representation)
        
        return y_pred, inter_attention_weights, attention_weight, attention_barcode, torch.cat(instance_preds, dim=0)
def save_file(threshold,data_list,label_list,epitopes,cluster_num,device,best_model,method,label_encoder1,X_name,dataset_name,it,lab,combo,max1,output_dir,output_dir1,save_inner,save_outer,save_outer_name,save_inner_name,save_score_name):
    
    encoded_epitopes = [label_encoder.transform(epi) for epi in epitopes]
    answer=list()
    num=0
    
    
    
    answer1=list()
    for data, label, epitope,sample in zip(data_list,label_list, encoded_epitopes,X_name):
        count_value=Counter(epitope)
        label_tensor = torch.tensor(epitope).to(device)
        barcode_tensor =torch.tensor(range(0,len(data))).to(device)
        for i in range(0,cluster_num):
            if Counter(epitope)[i]==0:
                new_row = np.zeros((1, 768))
                data=np.vstack((data, new_row.reshape(1,768)))
                count_value[i]=0.00001
                new_element = torch.tensor([i]).to(device)  # Ensure the new element is on the same device
                label_tensor = torch.cat((label_tensor, new_element))
                new_element = torch.tensor([-1]).to(device)
                barcode_tensor=torch.cat((barcode_tensor,new_element))
        for i in count_value.keys():
            if len(epitope)!=0:
                count_value[i]=count_value[i]/len(epitope)
        data_tensor = torch.tensor(np.array(data)).float().to(device)
        
        output = best_model(data_tensor, label_tensor,method,count_value,barcode_tensor)
        
        if save_inner==1:
            cluster_list=[]
            att_list=[]
            barcode_list=[]
            for cluster in range(len(output[2])):
                for barcode,att in zip(output[3][cluster],output[2][cluster]):
                    barcode_list.append(barcode.detach().cpu().numpy())
                    att_list.append(att.detach().cpu().numpy()[0])
                    cluster_list.append(label_encoder.inverse_transform([cluster])[0])
            df=np.transpose(pd.DataFrame([cluster_list,barcode_list,att_list]))
            df.columns=['epitope','barcode_index','attention_score']
            df.to_csv(output_dir+'/inter_attention/'+dataset_name+'/'+str(it)+'-'+sample+'_'.join(list(combo))+'.csv')
        
        if num==0:
            output_list1=output[1].detach().cpu().numpy()
            output_list=output[0].detach().cpu().numpy()
            num=1
        else:
            umap_all=output[1].detach().cpu().numpy()
            output_list1=np.hstack([output_list1,output[1].detach().cpu().numpy()])
            output_list=np.vstack([output_list,output[0].detach().cpu().numpy()])
    
        predictions = torch.argmax(output[0], dim=1).cpu().numpy()
        y_pred=label_encoder1.inverse_transform(predictions)
        answer.append(y_pred[0])
    
    df2=pd.DataFrame(output_list1)
    df2=np.transpose(df2)
    df2.index=X_name
    if save_inner==1:
        df2.to_csv(output_dir+'/'+save_outer_name+'/'+dataset_name+'_'+str(it)+'_'+max1+'_'+lab+'_'+str(cluster_num)+'_'+'_'.join(list(combo))+'.csv')
      
    df1=pd.DataFrame(output_list)
    df1.columns=list(label_encoder1.inverse_transform(range(0,2)))
    df1.index=X_name
    df=np.transpose(pd.DataFrame([answer,label_list]))
    df.columns=['Pred','Answer']
    df.index=X_name
    fbeta = f1_score(df.iloc[:,0],df.iloc[:,1], average = 'micro')
    if not os.path.exists(output_dir+'/'+save_score_name+'/'):
        os.makedirs(output_dir+'/'+save_score_name+'/')
    pd.DataFrame(df1).to_csv(output_dir+'/'+save_score_name+'/'+str(threshold)+'_'+dataset_name+'_'+'prop_'+str(it)+'_'+max1+'_'+lab+'_'+str(cluster_num)+'_'+'_'.join(list(combo))+'.csv')
    accuracy=roc_auc_score(label_encoder1.transform(label_list), df1[label_encoder1.inverse_transform([1])[0]])
    print(dataset_name)
    print(accuracy)
    return accuracy,fbeta
#def Data_loading(samples,input_dir,leng,list5,list8,list6,list9,clusternum,lab,sample_info,max1,weight_or_not):
def Data_loading(samples,input_dir,epitope_list,threshold,sample_info,lab,freq,frequency_or_not,length_or_not,length,vector_name,label_name,score_name,sample_info_dir):
    data_list=[]
    label_list=[]
    X=[]
    y=[]
    #BARCODE=[]
    epitopes=[]
    if length_or_not==1:
        if frequency_or_not==1:
            if freq=='reads':
                for sample in samples:
                    data=np.load(input_dir+'/'+vector_name+'/'+sample+'.npy')
                    label=pd.read_csv(input_dir+'/'+label_name+'/'+sample+'.csv')
                    #X1=pd.read_csv(input_dir+'/'+score_name+'/'+sample+'.csv')
                    reads=pd.read_csv('/'+sample_info_dir+'/'+sample+'.csv')
                    
                    label=label.iloc[:length,:]
                    reads=reads.iloc[:length,:]
                    data=data[:len(reads),:]
                    label1=np.array(reads[freq])
                    label1=np.array(reads[freq]/reads[freq].sum())
                    data=data*np.array(label1)[:, np.newaxis]
                    label=label[label['score']>=threshold]
                    data=data[label.index,:]
                    data_list.append(np.array(data))
                    label_list.append(sample_info.loc[sample,lab])
                    X.append(sample)
                    y.append(sample_info.loc[sample,lab])
                    epitopes.append(list(label.iloc[:,2]))
    if length_or_not==1:
        if frequency_or_not==1:
            if freq!='reads':
                for sample in samples:
                    data=np.load(input_dir+'/'+vector_name+'/'+sample+'.npy')
                    label=pd.read_csv(input_dir+'/'+label_name+'/'+sample+'.csv')
                    #=pd.read_csv(input_dir+'/'+score_name+'/'+sample+'.csv')
                    reads=pd.read_csv('/'+sample_info_dir+'/'+sample+'.csv')
                    label=label.iloc[:length,:]
                    reads=reads.iloc[:length,:]
                    data=data[:len(reads),:]
                    label1=np.array(reads[freq])
                    #label1=np.array(reads[freq])
                    data=data*np.array(label1)[:, np.newaxis]
                    label=label[label['score']>=threshold]
                    data=data[label.index,:]
                    data_list.append(np.array(data))
                    label_list.append(sample_info.loc[sample,lab])
                    X.append(sample)
                    y.append(sample_info.loc[sample,lab])
                    epitopes.append(list(label.iloc[:,2]))
    if length_or_not==1:
        if frequency_or_not==0:
            for sample in samples:
                data=np.load(input_dir+'/'+vector_name+'/'+sample+'.npy')
                label=pd.read_csv(input_dir+'/'+label_name+'/'+sample+'.csv')
                #X1=pd.read_csv(input_dir+'/'+score_name+'/'+sample+'.csv')
                reads=pd.read_csv('/'+sample_info_dir+'/'+sample+'.csv')
                label=label.iloc[:length,:]
                reads=reads.iloc[:length,:]
                data=data[:len(reads),:]
                label=label[label['score']>=threshold]
                data=data[label.index,:]
                data_list.append(np.array(data))
                label_list.append(sample_info.loc[sample,lab])
                X.append(sample)
                y.append(sample_info.loc[sample,lab])
                epitopes.append(list(label.iloc[:,2]))
    if length_or_not==0:
        if frequency_or_not==0:
            for sample in samples:
                data=np.load(input_dir+'/'+vector_name+'/'+sample+'.npy')
                label=pd.read_csv(input_dir+'/'+label_name+'/'+sample+'.csv')
                #X1=pd.read_csv(input_dir+'/'+score_name+'/'+sample+'.csv')
                #reads=pd.read_csv('/'+sample_info_dir+'/'+sample+'.csv')
                #data=data[:len(reads),:]
                label=label[label['score']>=threshold]
                data=data[label.index,:]
                data_list.append(np.array(data))
                label_list.append(sample_info.loc[sample,lab])
                X.append(sample)
                y.append(sample_info.loc[sample,lab])
                epitopes.append(list(label.iloc[:,2]))
    if length_or_not==0:
        if frequency_or_not==1:
            if freq=='reads':
                for sample in samples:
                    data=np.load(input_dir+'/'+vector_name+'/'+sample+'.npy')
                    label=pd.read_csv(input_dir+'/'+label_name+'/'+sample+'.csv')
                    #X1=pd.read_csv(input_dir+'/'+score_name+'/'+sample+'.csv')
                    reads=pd.read_csv('/'+sample_info_dir+'/'+sample+'.csv')
                    data=data[:len(reads),:]
                    label1=np.array(reads[freq])
                    label1=np.array(reads[freq]/reads[freq].sum())
                    data=data*np.array(label1)[:, np.newaxis]
                    label=label[label['score']>=threshold]
                    data=data[label.index,:]
                    data_list.append(np.array(data))
                    label_list.append(sample_info.loc[sample,lab])
                    X.append(sample)
                    y.append(sample_info.loc[sample,lab])
                    epitopes.append(list(label.iloc[:,2]))
    if length_or_not==0:
        if frequency_or_not==1:
            if freq!='reads':
                for sample in samples:
                    data=np.load(input_dir+'/'+vector_name+'/'+sample+'.npy')
                    label=pd.read_csv(input_dir+'/'+label_name+'/'+sample+'.csv')
                    #X1=pd.read_csv(input_dir+'/'+score_name+'/'+sample+'.csv')
                    reads=pd.read_csv('/'+sample_info_dir+'/'+sample+'.csv')
                    data=data[:len(reads),:]
                    label1=np.array(reads[freq])
                    #label1=np.array(reads[freq])
                    data=data*np.array(label1)[:, np.newaxis]
                    label=label[label['score']>=threshold]
                    data=data[label.index,:]
                    data_list.append(np.array(data))
                    label_list.append(sample_info.loc[sample,lab])
                    X.append(sample)
                    y.append(sample_info.loc[sample,lab])
                    epitopes.append(list(label.iloc[:,2]))
    return data_list,label_list,epitopes,X,y


##########

# Parameter input

#########

os.environ["CUDA_VISIBLE_DEVICES"]='0' # 0, 1, 2, 3 중 하나
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # GPU 점유 비율
session = InteractiveSession(config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

sample_info=pd.read_csv(metadata_dir+'/'+dataset_name+'.csv')
with open(metadata_dir+'/'+dataset_name+'_train.pkl', 'rb') as file:
    train_samples = pickle.load(file)
with open(metadata_dir+'/'+dataset_name+'_test.pkl', 'rb') as file:
    test_samples = pickle.load(file)
with open(metadata_dir+'/'+dataset_name+'_valid.pkl', 'rb') as file:
    valid_samples = pickle.load(file)
nCr = itertools.combinations(elements, 2)
for combo in list(nCr):
    print('_'.join(list(combo)))
    cross_valid=[]
    cross_test=[]
    
    save_score_name='label'

    Xx_list=list(range(0,106))
    epitope_list=Xx_list
    K=label_encoder.fit_transform(epitope_list)
    
    
    
    for it in range(0,1):
        sample_info=pd.read_csv(metadata_dir+'/'+dataset_name+'.csv')
        sample_info=sample_info.dropna(subset=[lab])
        sample_info.index=sample_info[sample_name_col]
        #sample_info=sample_info[sample_info[lab]!=list(set(elements)-set(list(combo)))[0]]
        train_data=list()
        train_label=list()
        train_epitopes=list()
        X_train=list()
        y_train=list()
        X_valid=list()
        y_valid=list()
        valid_data=list()
        valid_label=list()
        valid_epitopes=list()
        test_data=list()
        test_label=list()
        test_epitopes=list()
        X_test=list()
        y_test=list()
        sample_tensor=list()
        epitopes_tensor=list()
        sample_label_tensor=list()
        cell_type_tensor=list()
        data_tensor=list()
        train_data=list()
        
        ### Data Loading ####
        samples=list(set(train_samples[it])&set(sample_info.index))
       
        train_data,train_label,train_epitopes,X_train,y_train=Data_loading(samples,input_dir,epitope_list,threshold,sample_info,lab,freq,frequency_or_not,length_or_not,length,vector_name,label_name,score_name,sample_info_dir)
        samples=list(set(valid_samples[it])&set(sample_info.index))
        valid_data,valid_label,valid_epitopes,X_valid,y_valid=Data_loading(samples,input_dir,epitope_list,threshold,sample_info,lab,freq,frequency_or_not,length_or_not,length,vector_name,label_name,score_name,sample_info_dir)
        samples=list(set(test_samples[it])&set(sample_info.index))
        test_data,test_label,test_epitopes,X_test,y_test=Data_loading(samples,input_dir,epitope_list,threshold,sample_info,lab,freq,frequency_or_not,length_or_not,length,vector_name,label_name,score_name,sample_info_dir)
        
        epitope_list=Xx_list
        
        label_encoder.fit_transform(epitope_list)
        ### test
        methods =['attention']
        torch.set_num_threads(1)
        best=10
        num_break=0
        for method in methods:
            best=10
            num_break=0
            
            encoded_epitopes = [label_encoder.transform(epi) for epi in train_epitopes]
            label_encoder1 = LabelEncoder()
            TT=label_encoder1.fit_transform(list(combo))
            model = EpicPred(cluster_num, 2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, 
                                                                T_mult=1, eta_min=0.00001)
            weight=[]
            for i in range(0,2):
                total=len(sample_info.index)
                s=Counter(sample_info[lab])[label_encoder1.inverse_transform([i])[0]]
                weight.append(1-s/total)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(device))
            for epoch in range(epochs):
                num = 0
                model.train()
                answer1=list()
                umap_all1=list()
                encoded_epitopes = [label_encoder.transform(epi) for epi in train_epitopes]
                
                ### TRAIN
                
                
                for data, label, epitope in zip(train_data, train_label,encoded_epitopes):
                    count_value=Counter(epitope)
                    label_tensor = torch.tensor(epitope).to(device)
                    barcode_tensor =torch.tensor(range(0,len(data))).to(device)
                    for i in range(0,cluster_num):
                        if Counter(epitope)[i]==0:
                            new_row = np.zeros((1, 768))
                            data=np.vstack((data, new_row.reshape(1,768)))
                            count_value[i]=0.00001
                            new_element = torch.tensor([i]).to(device)  # Ensure the new element is on the same device
                            label_tensor = torch.cat((label_tensor, new_element))
                            new_element = torch.tensor([-1]).to(device)
                            barcode_tensor=torch.cat((barcode_tensor,new_element))
                    for i in count_value.keys():
                        if len(epitope)!=0:
                            count_value[i]=count_value[i]/len(epitope)
                    data_tensor = torch.tensor(np.array(data)).float().to(device)
                    optimizer.zero_grad()
                    output = model(data_tensor, label_tensor,method,count_value,barcode_tensor)
                    if num==0:
                        output_all=output[0]
                        umap_all1=output[1]
                        patch_pred=output[4]
                        labels=len(output[4])*[label]
                    else:
                        umap_all1=torch.cat([umap_all1,output[1]])
                        output_all=torch.cat([output_all,output[0]])
                        patch_pred=torch.cat((patch_pred,output[4]),axis=0)
                        labels1=len(output[4])*[label]
                        labels=labels+labels1
                    predictions = torch.argmax(output[0], dim=1).cpu().numpy()
                    y_pred=label_encoder1.inverse_transform(predictions)
                    num=1
                    answer1.append(y_pred[0])
                
                loss=criterion(output_all,F.one_hot(torch.tensor(label_encoder1.transform(list(train_label))).to(torch.int64)).to(device).float())
                loss1=criterion(patch_pred,F.one_hot(torch.tensor(label_encoder1.transform(list(labels))).to(torch.int64)).to(device).float())
                loss = 0.01*loss +1*loss1
                loss.backward()
                optimizer.step()
                print('train')
                print(loss)
                #print(loss)
                scheduler.step()
                model.eval()
                encoded_epitopes = [label_encoder.transform(epi) for epi in valid_epitopes]
                answer=list()
                answer1=list()
                umap_all=list()
                num1=0
                for data, label, epitope in zip(valid_data, valid_label, encoded_epitopes):
                    count_value=Counter(epitope)
                    label_tensor = torch.tensor(epitope).to(device)
                    barcode_tensor =torch.tensor(range(0,len(data))).to(device)
                    for i in range(0,cluster_num):
                        if Counter(epitope)[i]==0:
                            new_row = np.zeros((1, 768))
                            data=np.vstack((data, new_row.reshape(1,768)))
                            count_value[i]=0.00001
                            new_element = torch.tensor([i]).to(device)  # Ensure the new element is on the same device
                            label_tensor = torch.cat((label_tensor, new_element))
                            new_element = torch.tensor([-1]).to(device)
                            barcode_tensor=torch.cat((barcode_tensor,new_element))
                    for i in count_value.keys():
                        if len(epitope)!=0:
                            count_value[i]=count_value[i]/len(epitope)
                    data_tensor = torch.tensor(np.array(data)).float().to(device)
                    
                    optimizer.zero_grad()
                    output = model(data_tensor, label_tensor,method,count_value,barcode_tensor)
                    if num1==0:
                        output_list=output[0].detach().cpu().numpy()
                        umap_all=output[1]
                        output_all=output[0]
                        patch_pred=output[4]
                        labels=len(output[4])*[label]
                        num1=1
                    else:
                        umap_all=torch.cat([umap_all,output[1]])
                        output_all=torch.cat([output_all,output[0]])
                        output_list=np.vstack([output_list,output[0].detach().cpu().numpy()])
                        patch_pred=torch.cat((patch_pred,output[4]),axis=0)
                        labels1=len(output[4])*[label]
                        labels=labels+labels1
                    predictions = torch.argmax(output[0], dim=1).cpu().numpy()
                    y_pred=label_encoder1.inverse_transform(predictions)
                    answer.append(y_pred[0])
                df1=pd.DataFrame(output_list)
                df1.columns=list(label_encoder1.inverse_transform(range(0,2)))
                df1.index=X_valid
                
                loss2=criterion(patch_pred,F.one_hot(torch.tensor(label_encoder1.transform(list(labels))).to(torch.int64)).to(device).float())
                loss1=criterion(output_all,F.one_hot(torch.tensor(label_encoder1.transform(list(valid_label))).to(torch.int64)).to(device).float())
                loss1 = alpha*loss1 +beta*loss2


                accuracy=roc_auc_score(label_encoder1.transform(valid_label), df1[label_encoder1.inverse_transform([1])[0]])
                
                if loss1 < best:
                    best = loss1
                    best_model = copy.deepcopy(model)
                    best_accuracy=accuracy
                    print('valid')
                    print(loss1)
                    print(best_accuracy)
                    num_break=0
                    max1='Severe'
                    ac,fc=save_file(threshold,test_data,test_label,test_epitopes,cluster_num,device,best_model,method,label_encoder1,X_test,'test',it,lab,combo,max1,output_dir,output_dir1,save_inner,save_outer,save_outer_name,save_inner_name,save_score_name)
                    
                    print(ac)
                else:
                    num_break+=1
                    if num_break>=5:
                        break
                    
            ##### evaluation
            print(label_encoder.inverse_transform([0]))
            max1='Severe'
            ac,fc=save_file(threshold,train_data,train_label,train_epitopes,cluster_num,device,best_model,method,label_encoder1,X_train,'train',it,lab,combo,max1,output_dir,output_dir1,save_inner,save_outer,save_outer_name,save_inner_name,save_score_name)
            
            print(ac)
            ac,fc=save_file(threshold,valid_data,valid_label,valid_epitopes,cluster_num,device,best_model,method,label_encoder1,X_valid,'valid',it,lab,combo,max1,output_dir,output_dir1,save_inner,save_outer,save_outer_name,save_inner_name,save_score_name)
            
            print(ac)
            ac,fc=save_file(threshold,test_data,test_label,test_epitopes,cluster_num,device,best_model,method,label_encoder1,X_test,'test',it,lab,combo,max1,output_dir,output_dir1,save_inner,save_outer,save_outer_name,save_inner_name,save_score_name)
            print(ac)