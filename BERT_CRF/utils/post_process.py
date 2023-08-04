from transformers import BertTokenizer
from chemdataextractor.doc import Paragraph
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, SequentialSampler
import json
import torch
import numpy as np
from tqdm import tqdm
import re
import pandas as pd

class Post_processing:

    def __init__(self, modelname, file_name, abstract):
        self.tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case = False)
        self.file_name = file_name
        self.abstract = abstract

    def metadata(self):

        with open('./predict/dataset/'+ str(self.file_name)+'.json', "r", encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        metadata = json_data['metadata']
        metadata.pop('references')
        
        return metadata

    def sentence_list(self):
        para = Paragraph(self.abstract)
        sentences = para.raw_sentences
        
        token_list1 = []
        token_list2 = []
        number = 0
        for sentence in sentences:
            tmp = []
            tmp2 = []
            tokens = self.tokenizer.tokenize(sentence)

            for i in range(0,len(tokens)):
                token_loc = [tokens[i], number]
                tmp.append(token_loc)
                if tokens[i][:2] != '##':
                    tmp2.append(token_loc)
                number += 1

            token_list1.append(tmp)
            token_list2.append(tmp2)        

        return token_list1, token_list2

    def preprocess(self, token_list, prediction):

        token_list1, token_list2 = self.sentence_list()

        ner_token_list = []
        ner_token_list2 = []

        number = 0

        for corpus in token_list:
            tmp = []
            tmp2 = []
            for i in range(0,len(corpus)):
                if corpus[i] != '[CLS]' and corpus[i] != '[SEP]': 
                    token_loc = [corpus[i], number]
                    tmp.append(token_loc)
                    if corpus[i][:2] != '##':
                        tmp2.append(token_loc)
                    number += 1
                else:
                    token_loc = [corpus[i]]
                    tmp.append(token_loc)
                    tmp2.append(token_loc)

            ner_token_list.append(tmp)
            ner_token_list2.append(tmp2)  
        
        ner_token_list = sum(ner_token_list, [])
        ner_token_list2 = sum(ner_token_list2, [])
        prediction = sum(prediction, [])

        for i in range(0, len(prediction)):
            ner_token_list2[i].append(prediction[i])

        index_list = []
        for sentence in token_list1:
            sentence_index_list = []
            for i in range(0, len(sentence)):
                sentence_index_list.append(sentence[i][1])
            index_list.append(sentence_index_list)

        new_token_list = [] 
        for sentence in token_list1:

            sentence_first_index = sentence[0][-1]
            sentence_last_index = sentence[len(sentence)-1][-1]
            
            sentence_token = []
            for token in ner_token_list:
                if token[1] == 'O':
                    pass
                elif int(sentence_first_index) <= int(token[1]) and int(token[1]) <= int(sentence_last_index):
                    sentence_token.append(token)

            new_token_list.append(sentence_token)

        for i in range(0, len(new_token_list)):
            for j in range(0, len(new_token_list[i])):
                a = 1
                token = new_token_list[i][j]
                if token[0][:2] == '##' :
                    prev_number = token[1] - 1
                    prev_index = index_list[i].index(prev_number)
                    index_list[i][prev_index] = new_token_list[i][j][1]
                    index_list[i][j] = 'x'
                    tmp = [new_token_list[i][prev_index][0]+token[0][2:], new_token_list[i][j][1], new_token_list[i][prev_index][2]]
                    new_token_list[i][prev_index] = tmp
                    new_token_list[i][j] = ['x']

        final_list = []
        for sentence in new_token_list:
            token_list = []
            for token in sentence:
                if token[0] != 'x' :
                    token_list.append(token)
            final_list.append(token_list)

        return final_list

    def ner_predict(self, token_list, prediction):
        final_list = self.preprocess(token_list, prediction)

        result = []
        for i in range(0, len(final_list)):

            tmp = ''
            sentence_list = []
            entity_dict = {}

            for j in range(0, len(final_list[i])):
                token_tag = final_list[i][j][-1][0]
                token_entity = final_list[i][j][-1][2:]
                token = final_list[i][j][0]

                if token_tag == 'O':
                    if tmp != '':
                        sentence_list.append(tmp)
                    
                    tmp = ''    

                elif token_tag == 'B':
                    if tmp != '':
                        sentence_list.append(tmp)
                        
                    tmp = token_entity+'::'+token
            
                elif token_tag == 'I' :
                    if tmp != '':
                        tmp += ' ' + token
                
            for data in sentence_list :
                entity = data.split('::')[0]
                entity_name = data.split('::')[1]
                
                if entity in entity_dict:
                    value = entity_dict.get(entity)
                    entity_dict[entity] = [value , entity_name]
                else:
                    entity_dict[entity] = entity_name
            
            if entity_dict != {}:
                result.append(entity_dict)

        return result
    
    def ner_predict2(self, token_list, prediction):
        final_list = self.preprocess(token_list, prediction)

        result = []
        entity_dict = {}
        for i in range(0, len(final_list)):

            tmp = ''
            sentence_list = []
            # entity_dict = {}

            for j in range(0, len(final_list[i])):
                token_tag = final_list[i][j][-1][0]
                token_entity = final_list[i][j][-1][2:]
                token = final_list[i][j][0]

                if token_tag == 'O':
                    if tmp != '':
                        last_token_index = final_list[i][j-1][1]
                        sentence_list.append([tmp, (first_token_index + last_token_index)/2 ])
                    
                    tmp = ''    

                elif token_tag == 'B':
                    first_token_index = final_list[i][j][1]
                    if tmp != '':
                        if 'last_token_index' in locals()  :
                            sentence_list.append([tmp, (first_token_index + last_token_index)/2])
                        else: 
                            last_token_index = first_token_index
                        
                    tmp = token_entity+'::'+token
            
                elif token_tag == 'I' :
                    if tmp != '':
                        tmp += ' ' + token

            for data in sentence_list :
                entity = data[0].split('::')[0]
                entity_name = data[0].split('::')[1]
                entity_index = data[1]

                if entity in entity_dict:
                    value = entity_dict.get(entity)
                    value.append([entity_name, entity_index])
                    entity_dict[entity] = value
                else:
                    entity_dict[entity] = [[entity_name, entity_index]]

        print(entity_dict)

        return entity_dict










                

