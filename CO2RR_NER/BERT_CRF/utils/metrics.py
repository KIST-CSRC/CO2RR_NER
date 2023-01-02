import torch

def confusion_matrix(num_of_union, num_of_true, num_of_predict):

    if num_of_union == 0 :
        precision = 0
        recall = 0 
        f1_score = 0

    else:
        if num_of_true == 0 and num_of_predict == 0 :
            precision = 0
            recall = 0 
            f1_score = 0
        else:
            if num_of_true == 0:
                recall = 0
                precision = num_of_union / num_of_predict
                f1_score = 2 / (1/precision + 1/recall)
            elif num_of_predict == 0 :
                precision = 0
                recall = num_of_union / num_of_true
                f1_score = 2 / (1/precision + 1/recall)
            else:
                precision = num_of_union / num_of_predict
                recall = num_of_union / num_of_true                
                f1_score = 2 / (1/precision + 1/recall)

    return precision, recall, f1_score

def class_scores(entity_name, union, y_true, y_pred):
    # print(union)
    class_ = entity_name
    # print(f'{class_} precision    recall  f1-score\n')
    classes = set(union)
    
    if class_ in classes:
        class_p, class_r, class_f = confusion_matrix(
            union.count(class_), 
            y_true.count(class_), 
            y_pred.count(class_)
        )
    else :
        class_p = 0
        class_r = 0 
        class_f = 0

    avg_p, avg_r, avg_f = confusion_matrix(len(union), len(y_true), len(y_pred))
    
    return class_f, avg_f

def make_result_file(union, y_true, y_pred, out_txt, seed_number, k_fold):
    classes = set(union)
    
    with open(out_txt, 'a', encoding='UTF-8') as file:
        file.write(f'seed_number : {seed_number}, k_fold(test) : {k_fold}\n')

    for class_ in sorted(classes):
        # print(class_)
        p, r, f = confusion_matrix(
            union.count(class_), 
            y_true.count(class_), 
            y_pred.count(class_)
        )
        with open(out_txt, 'a', encoding='UTF-8') as file:
            file.write(f'{class_} f1: {f*100: 7.2f}% | {class_} precision: {p*100: 7.2f}% | {class_} recall: {r*100: 7.2f}% | \n')
    

    p, r, f = confusion_matrix(len(union), len(y_true), len(y_pred))
    class_ = 'structureO_micro_avg'

    with open(out_txt, 'a', encoding='UTF-8') as file:
        file.write(f'{class_} f1: {f*100: 7.2f}% | {class_} precision: {p*100: 7.2f}% | {class_} recall: {r*100: 7.2f}% | \n')

    union = [y.split('-')[0] for y in union if y.split('-')[0] != 'STRUCTURE']
    y_true = [y.split('-')[0] for y in y_true if y.split('-')[0] != 'STRUCTURE']
    y_pred = [y.split('-')[0] for y in y_pred if y.split('-')[0] != 'STRUCTURE']

    p, r, f = confusion_matrix(len(union), len(y_true), len(y_pred))
    class_ = 'structureX_micro_avg'

    with open(out_txt, 'a', encoding='UTF-8') as file:
        file.write(f'{class_} f1: {f*100: 7.2f}% | {class_} precision: {p*100: 7.2f}% | {class_} recall: {r*100: 7.2f}% | \n')

def preprocess(prediction_tags, label_tags):

    y_true = []
    y_pred = []

    for data in label_tags:
        for value in data:
            y_true.append(value)

    for data in prediction_tags:
        for value in data:
            y_pred.append(value)

        tmp = ''

    y_trues = []

    for i, y in enumerate(y_true):
        if y == 'O':
            if tmp != '':
                y_trues.append(tmp)
            
            tmp = ''
        
        elif y[0] == 'B' or y[0] == 'S':
            if tmp != '':
                y_trues.append(tmp)
                
            tmp = f'{y[2:]}-{i}'
            
        elif y[0] == 'I' or y[0] == 'E' or y[0] == 'X':
            if tmp != '':
                tmp += f'/{i}'

    tmp = ''

    y_preds = []

    for i, y in enumerate(y_pred):
        if y == 'O':
            if tmp != '':
                y_preds.append(tmp)
            
            tmp = ''
        
        elif y[0] == 'B' or y[0] == 'S':
            if tmp != '':
                y_preds.append(tmp)
                
            tmp = f'{y[2:]}-{i}'
            
        elif y[0] == 'I' or y[0] == 'E' or y[0] == 'X':
            if tmp != '':
                tmp += f'/{i}'
    

    # strict match

    union = [y for y in y_trues if y in y_preds]
    # print(len(union), len(y_trues), len(y_preds))

    union = [y.split('-')[0] for y in union]
    y_true = [y.split('-')[0] for y in y_trues]
    y_pred = [y.split('-')[0] for y in y_preds]

    return union, y_true, y_pred