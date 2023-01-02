import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.bert_model import BertCRFNERModel
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, SequentialSampler
from utils.data import NERData
import torch
import numpy as np
import random

import glob
import time

n_epochs = 1
full_finetuning = True
savename = time.strftime("%Y%m%d")

device = "cuda:1"
models = {'bert_base': 'bert-base-cased',
          'scibert': 'allenai/scibert_scivocab_cased',
          'matbert': './matbert-base-cased',
          'matscibert' : 'm3rg-iitd/matscibert',
          'bert_large' :'bert-large-cased'}

splits = {'_80_10_10': [0.8, 0.1, 0.1]}


label_type = 'BERT'
tag_type = 'IOB2'
# 0: exclude structure entities.
structure = '1'
batch_size = 32
k_fold = 10

datafile = f"./CO2RR_NER/dataset/new_500_{tag_type}_220701_matscibert.csv"

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

for alias, split in splits.items():
    
    for model_name in ['matscibert']:

        save_dir = os.getcwd()+'/{}_debug_{}_{}/'.format(model_name, tag_type, savename)

        ner_data = NERData(models[model_name])
        ner_data.preprocess(datafile, tag_type, structure)

        dataset, dataset_size, indice_dict = ner_data.k_fold_dataset(k_fold=10)
        classes = ner_data.classes

        for k in range(1, 21):
            test = 0
            seed_name = save_dir+'seed'+str(k)

            if not os.path.isdir(seed_name):
                os.makedirs(seed_name)

            test_result_name = seed_name+'/'+str(test)
            if not os.path.isdir(test_result_name):
                os.makedirs(test_result_name)

            train_indices = []
            for i in range(len(indice_dict)):
                if i == test :
                    test_indices = indice_dict[i]
                elif i == test+1 :
                    val_indices = indice_dict[i]
                else:
                    train_indices.extend(indice_dict[i])
            
            set_random_seed(k)

            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            set_random_seed(k)
            train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=train_sampler, pin_memory=True)
            test_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=test_sampler, pin_memory=True)
            val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=test_sampler, pin_memory=True)   

            set_random_seed(k)
            ner_model = BertCRFNERModel(modelname=models[model_name], classes=classes, device=device, lr=2e-4, results_file = test_result_name)
            set_random_seed(k)

            ner_model.train_eval(train_dataloader, val_dataloader=val_dataloader,test_dir = test_result_name, n_epochs=n_epochs, save_dir=save_dir, full_finetuning=full_finetuning, seed_number = k, k_fold = test)
            set_random_seed(k)
            ner_model.test(test_dataloader, test_dir = test_result_name, save_dir=save_dir, seed_number = k, k_fold = test)

            fs = glob.glob(save_dir+'epoch_*pt')

            for f in fs:
                try:
                    os.remove(f)
                except:
                    print('error while deleting file: {}'.format(f))
