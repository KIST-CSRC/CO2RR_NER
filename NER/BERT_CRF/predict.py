import sys
import os

from requests import post
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.bert_model import BertCRFNERModel
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, SequentialSampler
from utils.data import NERData
from utils.post_process import Post_processing
import torch
import numpy as np
import random
import pandas as pd

import glob
import time
import json

n_epochs = 1
full_finetuning = True
savename = time.strftime("%Y%m%d")

device = "cuda:3"
models = {'bert_base': 'bert-base-cased',
          'scibert': 'allenai/scibert_scivocab_cased',
          'matbert': './matbert-base-cased',
          'matscibert' : 'm3rg-iitd/matscibert',
          'bert_large' :'bert-large-cased'}

splits = {'_80_10_10': [0.8, 0.1, 0.1]}


tag_type = 'IOBES'
# 0: exclude structure entities.
structure = '1'

best_model_path = './best_model(CO2RR)/best.pt'
save_dir = best_model_path.split('/')[1]+'_'+best_model_path.split('/')[2]+'_'+best_model_path.split('/')[3]+'/'
datafile = f"./predict/dataset/Elsevier_CO2RR_00707.csv"

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

data = pd.read_csv(datafile)   

for model_name in ['matbert']:
    for i in range(0, len(data)):
        df = pd.read_csv(datafile)
        df = df.loc[[i]]
        file_name = df.loc[i]['file_name']
        abstract = df.loc[i]['abstract']

        ner_data = NERData(models[model_name])

        ner_data.preprocess(df, tag_type, structure, mode = 'predict')
        classes = ner_data.classes

        tensor_dataset = ner_data.dataset
        pred_dataloader = DataLoader(tensor_dataset) 

        ner_model = BertCRFNERModel(modelname=models[model_name], classes=classes, device=device, lr=2e-4, results_file = './')

        token_list, prediction_list = ner_model.predict(pred_dataloader, best_model_path)

        if tag_type == 'IOBE':
            for sentence in prediction_list:
                for i in range(0, len(sentence)):
                    if sentence[i][0] == 'E':
                        sentence[i] = 'I'+sentence[i][1:]
        
        elif tag_type == 'IOBES':
            for sentence in prediction_list:
                for i in range(0, len(sentence)):
                    if sentence[i][0] == 'E':
                        sentence[i] = 'I'+sentence[i][1:]
                    elif sentence[i][0] == 'S':
                        sentence[i] = 'B'+sentence[i][1:]

        post_process = Post_processing(models[model_name], file_name, abstract)

        metadata = post_process.metadata()
        year = metadata['year']
        # print(year)
        ner_predict = post_process.ner_predict2(token_list, prediction_list)
        
        json_dict = {}
        json_dict['metadata'] = metadata
        json_dict['ner_predict'] = ner_predict

        print(file_name+' success !')
        new_save_dir = './predict/220818/'+ save_dir

        if not os.path.isdir(new_save_dir):
            os.makedirs(new_save_dir)
        
        new_save_dir = new_save_dir + year + '/'

        if not os.path.isdir(new_save_dir):
            os.makedirs(new_save_dir)
            
        if not os.path.isfile(new_save_dir+file_name+'.json'):
            with open(new_save_dir+file_name+'.json', 'w', encoding='utf-8') as make_file:
                json.dump(json_dict, make_file, ensure_ascii=False, indent="\t")
        

        

