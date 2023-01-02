from pickle import NONE
from transformers import BertTokenizer, AutoTokenizer
from chemdataextractor.doc import Paragraph
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, SequentialSampler
import json
import torch
import numpy as np
from tqdm import tqdm
import re
import pandas as pd

class NERData:

    def __init__(self, modelname):
        self.tokenizer = AutoTokenizer.from_pretrained(modelname, strip_accents=False)
        self.dataset = None
        self.labels = None

    def load_from_csv(self, csv_file, except_class=['structure'], mode = None):


        if mode == 'predict':
            df = csv_file
            df.set_index('file_name')
        else:
            df = pd.read_csv(csv_file)
            df.set_index('file_name')

        RE_LISTSEP = re.compile('[\'\"], [\'\"]')
        RE_LISTELEMENT = re.compile('\[([^[]+)]')

        df['x_train'] = df['x_train'].map(lambda x: re.split(RE_LISTSEP, x[2:-2])) 
        df['y_train'] = df['y_train'].map(lambda x: re.split(RE_LISTSEP, x[2:-2]))
        
        file_name_list = df['file_name'].tolist()
    
        xs, ys = df['x_train'].tolist(), df['y_train'].tolist()

        if except_class:    
            for i, _ in enumerate(ys):
                for j, y in enumerate(_):
                    if sum([ec.upper() in y for ec in except_class]):
                        ys[i][j] = 'O'
                        
        empty_df = pd.DataFrame()

        for i in range(0,len(xs)):
            df = pd.DataFrame({'Word' : xs[i], 'Tag': ys[i]})
            df['file_name'] = np.nan
            df = df.fillna(file_name_list[i][:-4])
            empty_df = empty_df.append(df)
        
        empty_df['sentence'] = empty_df[['file_name','Word','Tag']].groupby(['file_name'])['Word'].transform(lambda x: ' '.join(x))
        empty_df['word_labels'] = empty_df[['file_name','Word','Tag']].groupby(['file_name'])['Tag'].transform(lambda x: ','.join(x))
        data = empty_df[["sentence", "word_labels", 'file_name']].drop_duplicates().reset_index(drop=True)
        
        return empty_df, data

    def preprocess(self, datafile, tag_type, structure, mode = None):

        data, data1 = self.load_from_csv(datafile, mode = mode)

        if tag_type == 'IOBES':
            classes = ['O', 'B-CATALYST', 'I-CATALYST', 'E-CATALYST', 'S-CATALYST',
                        'B-FE', 'I-FE', 'E-FE', 'S-FE', 
                        'B-CURRENT_DENSITY', 'I-CURRENT_DENSITY', 'E-CURRENT_DENSITY', 'S-CURRENT_DENSITY',
                        'B-REFERENCE_ELECTRODE', 'I-REFERENCE_ELECTRODE', 'E-REFERENCE_ELECTRODE', 'S-REFERENCE_ELECTRODE',
                        'B-CONCENTRATION', 'I-CONCENTRATION', 'E-CONCENTRATION', 'S-CONCENTRATION', 
                        'B-POTENTIAL', 'I-POTENTIAL', 'E-POTENTIAL', 'S-POTENTIAL',
                        'B-ELECTROLYTE', 'I-ELECTROLYTE', 'E-ELECTROLYTE', 'S-ELECTROLYTE',
                        'B-PRODUCT', 'I-PRODUCT', 'E-PRODUCT', 'S-PRODUCT',
                        'B-OVERPOTENTIAL', 'I-OVERPOTENTIAL', 'E-OVERPOTENTIAL', 'S-OVERPOTENTIAL',
                        'B-TOF', 'I-TOF', 'E-TOF', 'S-TOF', 
                        'B-STABILITY_HOUR', 'I-STABILITY_HOUR', 'E-STABILITY_HOUR', 'S-STABILITY_HOUR',
                        'B-STRUCTURE', 'I-STRUCTURE', 'E-STRUCTURE', 'S-STRUCTURE',
                        'B-ONSET_POTENTIAL', 'I-ONSET_POTENTIAL', 'E-ONSET_POTENTIAL', 'S-ONSET_POTENTIAL']
        elif tag_type == 'IOBE':
            classes = ['O', 'B-CATALYST', 'I-CATALYST', 'E-CATALYST', 
                        'B-FE', 'I-FE', 'E-FE', 
                        'B-CURRENT_DENSITY', 'I-CURRENT_DENSITY', 'E-CURRENT_DENSITY', 
                        'B-REFERENCE_ELECTRODE', 'I-REFERENCE_ELECTRODE', 'E-REFERENCE_ELECTRODE', 
                        'B-CONCENTRATION', 'I-CONCENTRATION', 'E-CONCENTRATION', 
                        'B-POTENTIAL', 'I-POTENTIAL', 'E-POTENTIAL', 
                        'B-ELECTROLYTE', 'I-ELECTROLYTE', 'E-ELECTROLYTE', 
                        'B-PRODUCT', 'I-PRODUCT', 'E-PRODUCT', 
                        'B-OVERPOTENTIAL', 'I-OVERPOTENTIAL', 'E-OVERPOTENTIAL', 
                        'B-TOF', 'I-TOF', 'E-TOF', 
                        'B-STABILITY_HOUR', 'I-STABILITY_HOUR', 'E-STABILITY_HOUR', 
                        'B-STRUCTURE', 'I-STRUCTURE', 'E-STRUCTURE', 
                        'B-ONSET_POTENTIAL', 'I-ONSET_POTENTIAL', 'E-ONSET_POTENTIAL']
        elif tag_type == 'IOB2':
            classes = ['O', 'B-CATALYST', 'I-CATALYST', 
                        'B-FE', 'I-FE', 
                        'B-CURRENT_DENSITY', 'I-CURRENT_DENSITY', 
                        'B-REFERENCE_ELECTRODE', 'I-REFERENCE_ELECTRODE', 
                        'B-CONCENTRATION', 'I-CONCENTRATION', 
                        'B-POTENTIAL', 'I-POTENTIAL',
                        'B-ELECTROLYTE', 'I-ELECTROLYTE', 
                        'B-PRODUCT', 'I-PRODUCT',
                        'B-OVERPOTENTIAL', 'I-OVERPOTENTIAL', 
                        'B-TOF', 'I-TOF', 
                        'B-STABILITY_HOUR', 'I-STABILITY_HOUR', 
                        'B-STRUCTURE', 'I-STRUCTURE', 
                        'B-ONSET_POTENTIAL', 'I-ONSET_POTENTIAL']
        if structure == '1':
            classes = [labels for labels in classes if labels[2:] != 'STRUCTURE'] 

        labels_to_ids = {k: v for v, k in enumerate(classes)}
        self.__get_iob_tags(classes)

        input_examples = []
        max_sequence_length = 512
        
        for i in range(0,len(data1)):         
            text = data1.sentence[i].split()
            labels = data1.word_labels[i].split(",")

            example = InputExample(i, text, labels)

            input_examples.append(example)

        features = self.__convert_examples_to_features(
                input_examples,
                self.classes,
                max_sequence_length,
        )

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)    
        
        dataset = TensorDataset(all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids)

        self.dataset = dataset
        return self

    def create_dataloaders(self, batch_size=30, train_frac=None, val_frac=0.1, dev_frac=0.1, shuffle_dataset=True):
        """
        Create train, val, and dev dataloaders from a preprocessed dataset
        Inputs:
            batch_size (int) :: Minibatch size for training
            train_frac (float or None) :: Fraction of data to use for training (None uses the remaining data)
            val_frac (float) :: Fraction of data to use for validation
            dev_frac (float) :: Fraction of data to use as a hold-out set
            shuffle_dataset (bool) :: Whether to randomize ordering of data samples
        Returns:
            dataloaders (tuple of torch.utils.data.Dataloader) :: train, val, and dev dataloaders
        """

        if self.dataset is None:
            print("No preprocessed dataset available")
            return None

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        dev_split = int(np.floor(dev_frac * dataset_size))
        val_split = int(np.floor(val_frac * dataset_size))+dev_split

        if shuffle_dataset :
            np.random.seed(105)
            np.random.shuffle(indices)

        dev_indices, val_indices = indices[:dev_split], indices[dev_split:val_split]

        if train_frac:
            train_split = int(np.floor(train_frac * dataset_size))+val_split
            train_indices = indices[val_split:train_split]
        else:
             train_indices = indices[val_split:]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SequentialSampler(val_indices)
        dev_sampler = SequentialSampler(dev_indices)

        self.train_dataloader = DataLoader(self.dataset, batch_size=batch_size,
            num_workers=0, sampler=train_sampler)

        if val_frac > 0:
            self.val_dataloader = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=0, sampler=val_sampler)
        else:
            self.val_dataloader = None

        if dev_frac > 0:
            self.dev_dataloader = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=0, sampler=dev_sampler)
        else:
            self.dev_dataloader = None

        return self.train_dataloader, self.val_dataloader, self.dev_dataloader

    def k_fold_dataset(self, shuffle_dataset=True, k_fold = None):
        indices_dict = {}

        if self.dataset is None:
            print("No preprocessed dataset available")
            return None

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))  
        fraction = 1/k_fold 
        subset = int(np.floor(fraction * dataset_size))

        if shuffle_dataset :
            np.random.seed(105)
            np.random.shuffle(indices)

        start = 0
        end = subset
        for i in range(k_fold):
            if i == 0 :
                indices_dict[i] = indices[:subset]
                start += subset
                end += subset 
            else :
                indices_dict[i] = indices[start:end]
                start += subset
                end += subset 

        return self.dataset, dataset_size, indices_dict

    def __convert_examples_to_features(
            self,
            examples,
            label_list,
            max_seq_length=512,
            cls_token_at_end=False,
            cls_token="[CLS]",
            cls_token_segment_id=1,
            sep_token="[SEP]",
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            pad_token_label_id=-100,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True,
            is_pretokenized = True
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        label_map = {label: i for i, label in enumerate(label_list)}
        span_labels = []
        for label in label_list:
            label = label.split('-')[-1]
            if label not in span_labels:
                span_labels.append(label)

        span_map = {label: i for i, label in enumerate(span_labels)}
        
        features = []

        example_range = tqdm(examples, desc='| writing examples |')

        for example in example_range:
            tokens = []
            valid_mask = []

            word_list = example.words
            # print(word_list)
            for i in range(0, len(word_list)):
                if is_pretokenized == True :
                    if word_list[i][0:2] != '##' :
                        valid_mask.append(1)
                    else:
                        valid_mask.append(0)

                    tokens.append(word_list[i])
    
                else:
                    word_tokens = self.tokenizer.tokenize(word_list[i])
                    for i, word_token in enumerate(word_tokens):
                        if i == 0:
                            valid_mask.append(1)
                        else:
                            valid_mask.append(0)

                        tokens.append(word_token)

            label_ids = [label_map[label] for label in example.labels]
     
            """
            #Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
            """

            entities = self.__get_entities(example.labels)

            start_ids = [span_map['O']] * len(label_ids)
            end_ids = [span_map['O']] * len(label_ids)

            for entity in entities:
                start_ids[entity[1]] = span_map[entity[0]]
                end_ids[entity[-1]] = span_map[entity[0]]
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2

            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]
                start_ids = start_ids[: (max_seq_length - special_tokens_count)]
                end_ids = end_ids[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            start_ids += [pad_token_label_id]
            end_ids += [pad_token_label_id]
            valid_mask.append(1)
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
                start_ids += [pad_token_label_id]
                end_ids += [pad_token_label_id]
                valid_mask.append(1)

            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                start_ids += [pad_token_label_id]
                end_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
                valid_mask.append(1)

            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                start_ids = [pad_token_label_id] + start_ids
                end_ids = [pad_token_label_id] + end_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                valid_mask.insert(0, 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)

            # pad token = 0
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                start_ids = ([pad_token_label_id] * padding_length) + start_ids
                end_ids = ([pad_token_label_id] * padding_length) + end_ids
                valid_mask = ([0] * padding_length) + valid_mask
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                # pad_token_segment_id = 0
                segment_ids += [pad_token_segment_id] * padding_length
                # pad_token_label_id = -100
                label_ids += [pad_token_label_id] * padding_length
                start_ids += [pad_token_label_id] * padding_length
                end_ids += [pad_token_label_id] * padding_length
                valid_mask += [0] * padding_length
            while (len(label_ids) < max_seq_length):
                label_ids.append(pad_token_label_id)
                start_ids.append(pad_token_label_id)
                end_ids.append(pad_token_label_id)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            try:
                assert len(label_ids) == max_seq_length
            except AssertionError:
                print(label_ids)
                print(len(label_ids), max_seq_length)
            assert len(start_ids) == max_seq_length
            assert len(end_ids) == max_seq_length
            assert len(valid_mask) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              valid_mask=valid_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              start_ids=start_ids,
                              end_ids=end_ids)
            )
        return features


    def __end_of_chunk(self, prev_tag, tag, prev_type, type_):
        """Checks if a chunk ended between the previous and current word.
        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.
        Returns:
            chunk_end: boolean.
        """
        chunk_end = False

        if prev_tag == 'E': chunk_end = True
        if prev_tag == 'S': chunk_end = True

        if prev_tag == 'B' and tag == 'B': chunk_end = True
        if prev_tag == 'B' and tag == 'S': chunk_end = True
        if prev_tag == 'B' and tag == 'O': chunk_end = True
        if prev_tag == 'I' and tag == 'B': chunk_end = True
        if prev_tag == 'I' and tag == 'S': chunk_end = True
        if prev_tag == 'I' and tag == 'O': chunk_end = True

        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
            chunk_end = True

        return chunk_end


    def __start_of_chunk(self, prev_tag, tag, prev_type, type_):
        """Checks if a chunk started between the previous and current word.
        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.
        Returns:
            chunk_start: boolean.
        """
        chunk_start = False

        if tag == 'B': chunk_start = True
        if tag == 'S': chunk_start = True

        if prev_tag == 'E' and tag == 'E': chunk_start = True
        if prev_tag == 'E' and tag == 'I': chunk_start = True
        if prev_tag == 'S' and tag == 'E': chunk_start = True
        if prev_tag == 'S' and tag == 'I': chunk_start = True
        if prev_tag == 'O' and tag == 'E': chunk_start = True
        if prev_tag == 'O' and tag == 'I': chunk_start = True

        if tag != 'O' and tag != '.' and prev_type != type_:
            chunk_start = True

        return chunk_start

    def __get_entities(self, seq):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
        """
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]

        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

            if self.__end_of_chunk(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i - 1))
            if self.__start_of_chunk(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_

        return set(chunks)

    def __collate_fn(self, batch):
        """
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
        max_len = batch_lens.max().item()
        results = ()
        for item in batch_tuple:
            if item.dim() >= 2:
                results += (item[:, :max_len],)
            else:
                results += (item,)
        return results

    def __get_iob_tags(self, labels):
        classes_raw = labels
        classes = ["O"]
        for c in classes_raw:
            if c == 'O':
                pass
            else:
                classes.append(c)
                # classes.append(c[2:])

        self.classes = classes

        return classes

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, valid_mask, segment_ids, label_ids, start_ids, end_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.valid_mask = valid_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.start_ids = start_ids
        self.end_ids = end_ids

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels

if __name__ == '__main__':
    datafile = "./dataset/nothing_IOB2_220415_BERT_bert_large_cased.csv"
    test = NERData('bert-base-cased')
    test.preprocess(datafile)
    # train_dataloader, val_dataloader, dev_dataloader = test.create_dataloaders()