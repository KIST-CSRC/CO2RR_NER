
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from utils.data import NERData
from itertools import product
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from utils.metrics import preprocess, class_scores, make_result_file
from seqeval.metrics import accuracy_score, f1_score
import torch.optim as optim
import numpy as np
import random
from abc import ABC, abstractmethod

class EarlyStopping:
    """ Stop training early if validation loss does not improve after given patience """
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint3.pt', result_file = None):
        """
        Args:
            patience (int): How long to wait after validation loss improves
            verbose (bool): If it is True, output message for improvement of each validation loss
            delta (float): Minimum change in monitered quantity that is considered improved
            path (str): checkpoint storage path
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.result_file = result_file

    def __call__(self, eval_f1, model):

        score = eval_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(eval_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            with open(self.result_file, "a+") as f:
                f.write(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(eval_f1, model)
            self.counter = 0

    def save_checkpoint(self, eval_f1, model):
        ''' Save the model when the validation loss decreases. '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {eval_f1:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = eval_f1

class NERModel(ABC):
    """
    A wrapper class for transformers models, implementing train, predict, and evaluate methods
    """

    def __init__(self, modelname="allenai/scibert_scivocab_cased", classes = ["O"], device="cuda:0", lr=5e-5, results_file=None):
        self.modelname = modelname
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.classes = classes
        self.config = AutoConfig.from_pretrained(modelname)
        self.config.num_labels = len(self.classes)
        self.config.model_name = self.modelname
        self.lr = lr
        self.device = device
        self.model = self.initialize_model()
        self.results_file = results_file

    def process_tags1(self, inputs, predicted, labels):
        labels_list = list(labels.cpu().numpy())

        batch_size, max_len = inputs['valid_mask'].shape
        valid_attention_mask = np.zeros((batch_size, max_len), dtype=int)
        for i in range(batch_size):


            for j in range(max_len):
                if inputs['valid_mask'][i][j].item() == 1:
                    if self.modelname == 'matbert' :
                        if inputs['input_ids'][i][j] not in (2, 3):
                            valid_attention_mask[i, j] = inputs['attention_mask'][i][j].item()
                    elif self.modelname == 'matscibert' :
                        if inputs['input_ids'][i][j] not in (102, 103):
                            valid_attention_mask[i, j] = inputs['attention_mask'][i][j].item()
                    elif self.modelname == 'scibert' or self.modelname == 'bert_base':
                        if inputs['input_ids'][i][j] not in (101, 102):
                            valid_attention_mask[i, j] = inputs['attention_mask'][i][j].item()
        
        valid_attention_mask = list(valid_attention_mask)
        valid_attention_mask1 = list(inputs['valid_mask'])

        prediction_tags = [[self.classes[i[j]] for j in range(0, len(i)) ] for i in predicted ]
        label_tags = [[self.classes[ii] if ii>=0 else self.classes[0] for ii, jj in zip(i, j) if jj==1] for i, j in zip(labels_list, valid_attention_mask1)]

        return prediction_tags, label_tags, valid_attention_mask

    def predict_process_tags(self, inputs, predicted):
        batch_size, max_len = inputs['valid_mask'].shape
        valid_attention_mask = np.zeros((batch_size, max_len), dtype=int)
        for i in range(batch_size):

            for j in range(max_len):
                if inputs['valid_mask'][i][j].item() == 1:
                    if self.modelname == 'matbert' :
                        if inputs['input_ids'][i][j] not in (2, 3):
                            valid_attention_mask[i, j] = inputs['attention_mask'][i][j].item()
                    elif self.modelname == 'matscibert' :
                        if inputs['input_ids'][i][j] not in (102, 103):
                            valid_attention_mask[i, j] = inputs['attention_mask'][i][j].item()
                    elif self.modelname == 'scibert' or self.modelname == 'bert_base':
                        if inputs['input_ids'][i][j] not in (101, 102):
                            valid_attention_mask[i, j] = inputs['attention_mask'][i][j].item()


        valid_attention_mask = list(valid_attention_mask)
        prediction_tags = [[self.classes[ii] for ii in i] for i in predicted ]
        
        return prediction_tags

    def train(self, train_dataloader, n_epochs, test_dir=None, val_dataloader=None, dev_dataloader = None, save_dir=None, full_finetuning=True, seed_number = None, k_fold = None):
        self.seed_number = seed_number
        self.save_dir = save_dir
        self.test_dir = test_dir
        patience=20
        verbose=False
        delta=0
        self.save_path = os.path.join(self.test_dir+'/', "best.pt")
        path= self.save_path
        result_file = test_dir+'/loss.txt'

        early_stopping = EarlyStopping(patience, verbose, delta, path, result_file)

        """
        Train the model
        Inputs:
            dataloader :: dataloader with training data
            n_epochs :: number of epochs to train
            val_dataloader :: dataloader with validation data - if provided the model with the best performance on the validation set will be saved
            save_dir :: directory to save models
        """
        self.val_f1_best = 0

        optimizer = self.create_optimizer(full_finetuning)
        scheduler = self.create_scheduler(optimizer, n_epochs, train_dataloader)

        epoch_metrics = {'training': {}, 'validation': {}}

        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)          

        for epoch in range(n_epochs):
            self.model.train()

            metrics = {'loss': [], 'catalyst_f1': [], 'micro_avg_f1': []}
            batch_range = tqdm(train_dataloader, desc='')

            epoch_cat = []
            epoch_f1 = []
            epoch_loss = []
            prediction_tags_all = []
            valid_tags_all = []

            for i, batch in enumerate(batch_range):

                inputs = {"input_ids": batch[0].to(self.device, non_blocking=True),
                          "attention_mask": batch[1].to(self.device, non_blocking=True),
                          "valid_mask": batch[2].to(self.device, non_blocking=True),
                          "labels": batch[4].to(self.device, non_blocking=True),
                          "decode": True}

                optimizer.zero_grad()
                loss, predicted = self.model.forward(**inputs)
                loss.backward()
 
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                labels = inputs['labels']

                prediction_tags, label_tags, valid_attention_mask = self.process_tags1(inputs, predicted, labels)

                epoch_loss.append(loss)
                prediction_tags_all.extend(prediction_tags)
                valid_tags_all.extend(label_tags)

                metrics['loss'].append(torch.mean(loss).item())
                union, y_true, y_pred = preprocess(prediction_tags, label_tags)

                catalyst_f1, micro_avg_f1 = class_scores('CATALYST',union, y_true, y_pred) 
                epoch_cat.append(catalyst_f1)
                epoch_f1.append(micro_avg_f1)
                metrics['catalyst_f1'].append(catalyst_f1)
                metrics['micro_avg_f1'].append(micro_avg_f1)
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]
                self.my_lr = scheduler.optimizer.param_groups[0]['lr']

                batch_range.set_description('| training | epoch: {:d}/{:d} | loss: {:.4f} | catalyst_f1: {:.4f} | micro_avg f1: {:.4f} |'.format(epoch+1, n_epochs, *means))
            
            epoch_loss = torch.mean(torch.stack(epoch_loss)).item()
            epoch_cat= np.mean(epoch_cat)
            epoch_f1 = np.mean(epoch_f1)

            with open(test_dir+'/loss.txt', "a+") as f:
                f.write("n_epochs : {}, lr : {}, train_loss : {}, train_cat_f1: {}, train_f1_score : {}\n".format(epoch, self.my_lr, epoch_loss, epoch_cat, epoch_f1))
            
            epoch_metrics['training']['epoch_{}'.format(epoch)] = metrics

            early_stopping(epoch_f1, self.model)

            if early_stopping.early_stop:
                with open(test_dir+'/loss.txt', "a+") as f:
                    f.write("Early stopping at "+str(epoch)+'\n')
                self.early_stop = True
                break
        
        history_save_path = os.path.join(self.test_dir+'/', 'history.pt')
        torch.save(epoch_metrics, history_save_path)

        return 

    def train_eval(self, train_dataloader, n_epochs, test_dir=None, val_dataloader=None, dev_dataloader = None, save_dir=None, full_finetuning=True, seed_number = None, k_fold = None):
        self.seed_number = seed_number
        self.save_dir = save_dir
        self.test_dir = test_dir
        patience=20
        verbose=False
        delta=0
        self.save_path = os.path.join(self.test_dir+'/', "best.pt")
        path= self.save_path
        result_file = self.results_file+'/loss.txt'

        early_stopping = EarlyStopping(patience, verbose, delta, path, result_file)
        # print(self.seed_number)
        """
        Train the model
        Inputs:
            dataloader :: dataloader with training data
            n_epochs :: number of epochs to train
            val_dataloader :: dataloader with validation data - if provided the model with the best performance on the validation set will be saved
            save_dir :: directory to save models
        """
        self.val_f1_best = 0

        optimizer = self.create_optimizer(full_finetuning)
        scheduler = self.create_scheduler(optimizer, n_epochs, train_dataloader)

        epoch_metrics = {'training': {}, 'validation': {}}

        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)          

        for epoch in range(n_epochs):
            self.model.train()

            metrics = {'loss': [], 'catalyst_f1': [], 'micro_avg_f1': []}
            batch_range = tqdm(train_dataloader, desc='')

            epoch_cat = []
            epoch_f1 = []
            epoch_loss = []
            prediction_tags_all = []
            valid_tags_all = []

            for i, batch in enumerate(batch_range):

                inputs = {"input_ids": batch[0].to(self.device),
                          "attention_mask": batch[1].to(self.device),
                          "valid_mask": batch[2].to(self.device),
                          "labels": batch[4].to(self.device),
                          "decode": True}

                optimizer.zero_grad()
                loss, predicted = self.model.forward(**inputs)

                loss.backward()
               
                optimizer.step()
                scheduler.step()

                labels = inputs['labels']

                prediction_tags, label_tags, valid_attention_mask = self.process_tags1(inputs, predicted, labels)

                epoch_loss.append(loss)
                prediction_tags_all.extend(prediction_tags)
                valid_tags_all.extend(label_tags)

                # catalyst 성능 표시와 f1_score(strict) 우리 라벨링에 맞게 변환해서 코드 짜기
                metrics['loss'].append(torch.mean(loss).item())
                union, y_true, y_pred = preprocess(prediction_tags, label_tags)
                # print(union)
                catalyst_f1, micro_avg_f1 = class_scores('CATALYST',union, y_true, y_pred) 
                epoch_cat.append(catalyst_f1)
                epoch_f1.append(micro_avg_f1)
                metrics['catalyst_f1'].append(catalyst_f1)
                metrics['micro_avg_f1'].append(micro_avg_f1)
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]
                self.my_lr = scheduler.optimizer.param_groups[0]['lr']

                batch_range.set_description('| training | epoch: {:d}/{:d} | loss: {:.4f} | catalyst_f1: {:.4f} | micro_avg f1: {:.4f} |'.format(epoch+1, n_epochs, *means))
            
            epoch_loss = torch.mean(torch.stack(epoch_loss)).item()
            epoch_cat= np.mean(epoch_cat)
            epoch_f1 = np.mean(epoch_f1)

            with open(test_dir+'/loss.txt', "a+") as f:
                f.write("n_epochs : {}, lr : {}, train_loss : {}, train_cat_f1: {}, train_f1_score : {}\n".format(epoch, self.my_lr, epoch_loss, epoch_cat, epoch_f1))
            
            epoch_metrics['training']['epoch_{}'.format(epoch)] = metrics


            _, _, eval_f1 = self.evaluate(val_dataloader, test_dir = self.test_dir, lr=self.lr, n_epochs=epoch, validate=True, save_path=os.path.join(save_dir, "best.pt"),  seed_number = self.seed_number)
            early_stopping(eval_f1, self.model)

            if early_stopping.early_stop:
                with open(test_dir+'/loss.txt', "a+") as f:
                    f.write("Early stopping at "+str(epoch)+'\n')
                self.early_stop = True
                break
        
        history_save_path = os.path.join(self.test_dir+'/', 'history.pt')
        torch.save(epoch_metrics, history_save_path)

        return 

    def load_model(self,save_path):
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        return

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def create_scheduler(self, optimizer, n_epochs, train_dataloader):
        pass

    def evaluate(self, dataloader, test_dir=None, validate=False, save_path=None, lr=None, n_epochs=None, seed_number = None):
        mode = 'validation'
        self.model.eval()

        eval_loss = []
        eval_pred = []
        eval_label = []
        prediction_tags_all = []
        valid_tags_all = []
        eval_cat = []
        eval_f1 = []
        metrics = {'loss': [], 'catalyst_f1': [],  'micro_avg_f1': []}
        batch_range = tqdm(dataloader, desc='')

        with torch.no_grad():
            for batch in batch_range:
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device),
                    "decode": True
                }

                loss, predicted = self.model.forward(**inputs)
                labels = inputs['labels']

                eval_loss.append(loss)
                eval_pred.append(predicted)
                eval_label.append(labels)

                prediction_tags, label_tags, valid_attention_mask = self.process_tags1(inputs, predicted, labels)

                prediction_tags_all.extend(prediction_tags)
                valid_tags_all.extend(label_tags)

                metrics['loss'].append(torch.mean(loss).item())
                union, y_true, y_pred = preprocess(prediction_tags, label_tags)
                catalyst_f1, micro_avg_f1 = class_scores('CATALYST',union, y_true, y_pred) 
                eval_cat.append(catalyst_f1)
                eval_f1.append(micro_avg_f1)
                metrics['catalyst_f1'].append(catalyst_f1)
                metrics['micro_avg_f1'].append(micro_avg_f1)
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]

                batch_range.set_description('| {} (rolling average) | loss: {:.4f} | catalyst_f1: {:.4f} | micro_avg f1: {:.4f} |'.format(mode, *means)) 

        eval_loss = torch.mean(torch.stack(eval_loss)).item()
        eval_cat= np.mean(eval_cat)
        eval_f1 = np.mean(eval_f1)
     
        with open(self.test_dir+'/loss.txt', "a+") as f:
            f.write("n_epochs : {}, lr : {}, eval_loss : {}, eval_cat_f1: {}, eval_f1_score : {}\n".format(n_epochs, self.my_lr, eval_loss, eval_cat, eval_f1))
        
        print("| {} (epoch evaluation) | loss: {:.4f} | catalyst_f1: {:.4f} | micro_avg f1: {:.4f} |".format(mode, eval_loss, eval_cat, eval_f1))
        
        return eval_loss, eval_cat, eval_f1

    def test(self, dataloader, validate=False, save_dir=None, lr=None, test_dir=None, n_epochs=None, seed_number = None, k_fold = None):
        
        save_path = os.path.join(test_dir, "best.pt")
        self.model.load_state_dict(torch.load(str(save_path)))
        
        self.model.eval()
        mode = 'test'

        eval_loss = []
        eval_pred = []
        eval_label = []
        test_prediction_tags_all = []
        test_valid_tags_all = []
        test_input_ids = []
        total_catalyst = []
        total_f1 = []
        metrics = {'loss': [], 'catalyst_f1': [],  'micro_avg_f1': []}
        batch_range = tqdm(dataloader, desc='')

        with torch.no_grad():
            for batch in batch_range:
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device),
                    "decode": True
                }

                loss, predicted = self.model.forward(**inputs)
                labels = inputs['labels']
                

                eval_loss.append(loss)
                eval_pred.append(predicted)
                eval_label.append(labels)

                prediction_tags, label_tags, valid_attention_mask = self.process_tags1(inputs, predicted, labels)

                token_id = inputs['input_ids']
                print(token_id[0])

                for i in range(0,len(token_id)):
                    test_input_ids.append(token_id[i])
                test_prediction_tags_all.extend(prediction_tags)
                test_valid_tags_all.extend(label_tags)

                metrics['loss'].append(torch.mean(loss).item())
                union, y_true, y_pred = preprocess(prediction_tags, label_tags)
                catalyst_f1, micro_avg_f1 = class_scores('CATALYST',union, y_true, y_pred) 
                total_catalyst.append(catalyst_f1)
                total_f1.append(micro_avg_f1)
                metrics['catalyst_f1'].append(catalyst_f1)
                metrics['micro_avg_f1'].append(micro_avg_f1)
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]
                
                batch_range.set_description('| {} (rolling average) | loss: {:.4f} | catalyst_f1: {:.4f} | micro_avg f1: {:.4f} |'.format(mode, *means))
            # print(test_prediction_tags_all)
            eval_loss = torch.mean(torch.stack(eval_loss)).item()
            eval_cat_f1 = np.mean(total_catalyst)
            eval_f1_score = np.mean(total_f1)
        
        if validate == False:
            with open(test_dir+'/loss.txt', "a+") as f:
                f.write("lr : {}, test_loss : {}, test_cat_f1: {}, test_f1_score : {}\n".format(self.my_lr, eval_loss, eval_cat_f1, eval_f1_score))

            token_list  = []
            for i in range(0, len(test_input_ids)):
                paper_token = []
                for token in self.tokenizer.convert_ids_to_tokens(test_input_ids[i]):

                    if token != '[PAD]' and token[0:2] != '##':
                        paper_token.append(token)

                token_list.append(paper_token)

            if self.save_dir is not None and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir) 

            for i in range(0, len(test_prediction_tags_all)):

                with open(test_dir+'/visual.txt', "a") as f:
                    f.write("token's number : "+str(len(test_prediction_tags_all[i]))+'\n')
                for j in range(0, len(token_list[i])):
                    with open(test_dir+'/visual.txt', "a") as f:
                        f.write(str(token_list[i][j])+' '+str(test_valid_tags_all[i][j])+' '+str(test_prediction_tags_all[i][j])+'\n')
            
            union, y_true, y_pred = preprocess(test_prediction_tags_all, test_valid_tags_all)
            make_result_file(union, y_true, y_pred, self.save_dir+'summary.txt', seed_number, k_fold)
        
        print("| {} (epoch evaluation) | loss: {:.4f} | catalyst_f1: {:.4f} | micro_avg f1: {:.4f} |".format(mode, eval_loss, eval_cat_f1, eval_f1_score))
        
        return metrics


    def predict(self, data, trained_model=None, labels=None):
        self.model.eval()
        self.labels = labels
        # print(len(self.labels))

        self.model.load_state_dict(torch.load(trained_model))

        test_prediction_tags_all = []
        test_input_ids = []

        # run predictions
        with torch.no_grad():
            for i, batch in enumerate(data):

                # get masked inputs and run predictions
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device),
                    "decode": True
                }
                loss, predicted = self.model.forward(**inputs)

                prediction_tags = self.predict_process_tags(inputs, predicted)

                token_id = inputs['input_ids']
                for i in range(0,len(token_id)):
                    test_input_ids.append(token_id[i])
                test_prediction_tags_all.extend(prediction_tags)
 
            token_list  = []
            for i in range(0, len(test_input_ids)):
                paper_token = []
                for token in self.tokenizer.convert_ids_to_tokens(test_input_ids[i]):
                    if token != '[PAD]' :
                        paper_token.append(token)
                token_list.append(paper_token)

        return token_list, test_prediction_tags_all

    def _data_to_dataloader(self, data):
        # check for input data type
        if os.path.isfile(data):
            texts = self.load_file(data)
        elif type(data) == list:
            texts = data
        elif type(data) == str:
            texts = [data]
        else:
            print("Please provide text or set of texts (directly or in a file path format) to predict on!")

        # tokenize and preprocess input data
        if self.labels:
            labels = self.labels
        else:
            labels = []
            for label in self.classes:
                if label != 'O' and label[2:] not in labels:
                    labels.append(label[2:])
        ner_data = NERData(modelname=self.modelname)
        tokenized_dataset = []
        for text in texts:
            tokenized_text = ner_data.create_tokenset(text)
            tokenized_text['labels'] = labels
            tokenized_dataset.append(tokenized_text)
        ner_data.preprocess(tokenized_dataset,is_file=False)
        tensor_dataset = ner_data.dataset
        pred_dataloader = DataLoader(tensor_dataset)

        return tokenized_dataset, pred_dataloader

if __name__ == '__main__':
    test = NERModel()
    