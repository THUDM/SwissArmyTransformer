
import numpy as np
import os
import torch
import math
from torch.optim import Optimizer
from torch.distributions.bernoulli import Bernoulli
from typing import Callable, Iterable, Tuple
from tqdm import tqdm
from SwissArmyTransformer.training.utils import Timers
from SwissArmyTransformer import mpu

pretrain_path = ''
from transformers import RobertaTokenizer
from transformers import DebertaTokenizer
from transformers import BertTokenizer
tokenizers = {}
tokenizers['roberta-large'] = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))
tokenizers['roberta-large-addprefix'] = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'), add_prefix_space=False)

# tokenizers['bert-large'] = BertTokenizer.from_pretrained(os.path.join(pretrain_path, 'bert-large-uncased'), local_files_only=False)
# tokenizers['bert-large-addprefix'] = BertTokenizer.from_pretrained(os.path.join(pretrain_path, 'bert-large-uncased'), local_files_only=True, add_prefix_space=False)

# tokenizers['deberta-large'] = DebertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'microsoft/deberta-large'), local_files_only=True)
jishu = [0,0]
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
from SwissArmyTransformer.data_utils import load_hf_dataset

def get_dataset_keys(dataset_name):
    dataset_keys = {
        #superglue
        'boolq':["input_ids", "position_ids", "attention_mask", "label"],
        'rte':["input_ids", "position_ids", "attention_mask", "label"],
        'copa':["input_ids_1", "position_ids_1", "attention_mask_1", "input_ids_2", "position_ids_2", "attention_mask_2", "label"],
        'cb':["input_ids", "position_ids", "attention_mask", "label"],
        'wic':["input_ids", "position_ids", "attention_mask", "label", "pos1", "pos2", "pos1_e", "pos2_e"],
        'multirc':["input_ids", "position_ids", "attention_mask", "label"],
        'wsc': ["label", "query_tokens", "query_mask", "cand_tokens", "cand_mask"],
        #glue
        'mrpc':["input_ids", "position_ids", "attention_mask", "label"],
        'qnli':["input_ids", "position_ids", "attention_mask", "label"],
        'qqp':["input_ids", "position_ids", "attention_mask", "label"],
        'cola':["input_ids", "position_ids", "attention_mask", "label"],
        'wnli':["input_ids", "position_ids", "attention_mask", "label"],
        'sst2':["input_ids", "position_ids", "attention_mask", "label"],
        'stsb':["input_ids", "position_ids", "attention_mask", "stsb_label"],
        #squad
        'squad':["input_ids", "position_ids", "attention_mask", "label", "start_list", "end_list"],
        'squad_v2':["input_ids", "position_ids", "attention_mask", "label", "start_list", "end_list"],
        #conll
        'conll2003':["input_ids", "position_ids", "attention_mask", "label"],
        #emotion
        'emotion':["input_ids", "position_ids", "attention_mask", "label"],
        #semeval2014
        'semeval2014':["input_ids", "position_ids", "attention_mask", "label", "pos", "pos_e"]
    }
    return dataset_keys[dataset_name]

def get_class_num(dataset_name):
    dataset_class_num = {
        'boolq':2,
        'rte':2,
        'copa':2,
        'cb':3,
        'wic':2,
        'mrpc':2,
        'qnli':2,
        'qqp':2,
        'cola':2,
        'wnli':2,
        'sst2':2,
        'stsb':2,
        'squad':2,
        'squad_v2':2,
        'conll2003':9,
        'emotion':6,
        'multirc':2,
        'wsc':2,
        'semeval2014':3,
    }
    return dataset_class_num[dataset_name]

def get_batch_function(dataset_name):
    if dataset_name == "semeval2014" :
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ['input_ids', 'position_ids', 'attention_mask', 'label', 'pos', 'pos_e']
            datatype = torch.int64

            # Broadcast data.
            timers('data loader').start()
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            timers('data loader').stop()
            data_b = mpu.broadcast_data(keys, data, datatype)
            # Unpack.
            tokens = data_b['input_ids'].long()
            labels = data_b['label'].long()
            position_ids = data_b['position_ids'].long()
            attention_mask = data_b['attention_mask'][:, None, None, :].float()
            pos = data_b['pos'].long()
            pos_e = data_b['pos_e'].long()

            bz = labels.shape[0]
            word = torch.zeros_like(tokens, dtype=attention_mask.dtype, device=tokens.device)
            for i in range(bz):
                len = (pos_e[i] - pos[i] + 1)
                for j in range(pos[i],pos_e[i]+1):
                    word[i,j] = 1/len

            bz = tokens.shape[0]
            for i in range(bz):
                pos[i] += i * tokens.shape[1]
            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()
                word = word.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1), {'pos':pos, 'word':word}
    elif dataset_name == "stsb":
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ['input_ids', 'position_ids', 'attention_mask', 'stsb_label']
            datatype = torch.int64

            # Broadcast data.
            timers('data loader').start()
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            timers('data loader').stop()
            data_b = mpu.broadcast_data(keys, data, datatype)
            # Unpack.
            tokens = data_b['input_ids'].long()
            labels = data_b['stsb_label'].long()
            position_ids = data_b['position_ids'].long()
            attention_mask = data_b['attention_mask'][:, None, None, :].float()

            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1)
    elif dataset_name in ["boolq", "rte", "cb", "mrpc", "qnli", "qqp", "cola", 'wnli', 'conll2003', 'sst2', 'emotion', 'multirc', 'stsb'] :
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ['input_ids', 'position_ids', 'attention_mask', 'label']
            datatype = torch.int64

            # Broadcast data.
            timers('data loader').start()
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            timers('data loader').stop()
            data_b = mpu.broadcast_data(keys, data, datatype)
            # Unpack.
            tokens = data_b['input_ids'].long()
            labels = data_b['label'].long()
            position_ids = data_b['position_ids'].long()
            attention_mask = data_b['attention_mask'][:, None, None, :].float()

            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1)
    elif dataset_name == "wsc":
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ["label", "query_tokens", "query_mask", "cand_tokens", "cand_mask", "label"]
            datatype = torch.int64

            # Broadcast data.
            timers('data loader').start()
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            timers('data loader').stop()
            data_b = mpu.broadcast_data(keys, data, datatype)
            # Unpack.
            query_tokens = data_b['query_tokens'].long()
            query_mask = data_b['query_mask'].long()
            cand_tokens = data_b['cand_tokens'].long()
            cand_mask = data_b['cand_mask'].long()
            labels = data_b['label'].long()

            query_mask = query_mask.bool()
            cand_mask = cand_mask.bool()

            cand_tokens = cand_tokens.reshape([query_tokens.shape[0], -1, query_tokens.shape[1]])
            cand_mask = cand_mask.reshape([query_mask.shape[0], -1, query_mask.shape[1]])
            padding_idx = 1
            query_position_ids = create_position_ids_from_input_ids(query_tokens, 1, 0)
            query_attention_mask = (torch.ones_like(query_position_ids, device=query_tokens.device) * query_position_ids.ne(padding_idx))[:,None,None,:].float()
            cand_tokens_position_ids = create_position_ids_from_input_ids(cand_tokens.reshape([cand_tokens.shape[0]*cand_tokens.shape[1],-1]), 1, 0).reshape(cand_tokens.shape)
            cand_attention_mask = (torch.ones_like(cand_tokens_position_ids, device=query_tokens.device) * cand_tokens_position_ids.ne(padding_idx))[:,:,None,None,:].float()

            # Convert
            if args.fp16:
                query_attention_mask = query_attention_mask.half()
                cand_attention_mask = cand_attention_mask.half()
            return query_tokens, query_mask, query_attention_mask, query_position_ids, cand_tokens, cand_mask, cand_attention_mask, cand_tokens_position_ids, labels
    elif dataset_name == "copa":
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ["input_ids_1", "position_ids_1", "attention_mask_1", "input_ids_2", "position_ids_2", "attention_mask_2", "label"]
            datatype = torch.int64

            # Broadcast data.
            timers('data loader').start()
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            timers('data loader').stop()
            data_b = mpu.broadcast_data(keys, data, datatype)
            # Unpack.
            tokens_1 = data_b['input_ids_1'].long()
            tokens_2 = data_b['input_ids_2'].long()
            tokens = torch.cat([tokens_1, tokens_2], dim=0)
            labels = data_b['label'].long()
            position_ids_1 = data_b['position_ids_1'].long()
            position_ids_2 = data_b['position_ids_2'].long()
            position_ids = torch.cat([position_ids_1, position_ids_2], dim=0)

            attention_mask_1 = data_b['attention_mask_1'][:, None, None, :].float()
            attention_mask_2 = data_b['attention_mask_2'][:, None, None, :].float()
            attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=0)

            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1)
    elif dataset_name == 'wic':
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ['input_ids', 'position_ids', 'attention_mask', 'label', 'pos1', 'pos2', 'pos1_e', 'pos2_e']
            datatype = torch.int64

            # Broadcast data.
            timers('data loader').start()
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            timers('data loader').stop()
            data_b = mpu.broadcast_data(keys, data, datatype)
            # Unpack.
            tokens = data_b['input_ids'].long()
            labels = data_b['label'].long()
            position_ids = data_b['position_ids'].long()
            attention_mask = data_b['attention_mask'][:, None, None, :].float()
            pos1 = data_b['pos1'].long()
            pos2 = data_b['pos2'].long()
            pos1_e = data_b['pos1_e'].long()
            pos2_e = data_b['pos2_e'].long()

            bz = labels.shape[0]
            word1 = torch.zeros_like(tokens, dtype=attention_mask.dtype, device=tokens.device)
            word2 = torch.zeros_like(tokens, dtype=attention_mask.dtype, device=tokens.device)
            for i in range(bz):
                len = (pos1_e[i] - pos1[i] + 1)
                for j in range(pos1[i],pos1_e[i]+1):
                    word1[i,j] = 1/len
                len = (pos2_e[i] - pos2[i] + 1)
                for j in range(pos2[i], pos2_e[i]+1):
                    word2[i,j] = 1/len

            bz = tokens.shape[0]
            for i in range(bz):
                pos1[i] += i * tokens.shape[1]
                pos2[i] += i * tokens.shape[1]

            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()
                word1 = word1.half()
                word2 = word2.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1), {'pos1':pos1, 'pos2':pos2, 'word1':word1, 'word2':word2}
    elif dataset_name in ['squad', 'squad_v2']:
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ['input_ids', 'position_ids', 'attention_mask', 'label', 'start_list', 'end_list']
            datatype = torch.int64

            # Broadcast data.
            timers('data loader').start()
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            timers('data loader').stop()
            data_b = mpu.broadcast_data(keys, data, datatype)
            # Unpack.
            tokens = data_b['input_ids'].long()
            labels = data_b['label'].long()
            position_ids = data_b['position_ids'].long()
            attention_mask = data_b['attention_mask'][:, None, None, :].float()
            start_list = data_b['start_list'].long()
            end_list = data_b['end_list'].long()
            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1), {'start_list':start_list, 'end_list':end_list}
    else:
        raise Exception('dataset name is wrong')
    return get_batch

def _encode_single_text(text, args, is_split_into_words=False):
    if is_split_into_words == True:
        tokenizer_name = args.name_model+"-addprefix"
    else:
        tokenizer_name = args.name_model
    encoded_input = tokenizers[tokenizer_name](text, max_length=args.sample_length, padding='max_length', truncation='only_first', is_split_into_words=is_split_into_words)
    # if 'roberta' in args.model_type:
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    # else:
    #     position_ids =
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

def _encode_double_text(text, text_pair, args, truncation='only_first', is_split_into_words=False):
    if is_split_into_words == True:
        tokenizer_name = args.name_model+"-addprefix"
    else:
        tokenizer_name = args.name_model
    encoded_input = tokenizers[tokenizer_name](text, text_pair, max_length=args.sample_length, padding='max_length', truncation=truncation, is_split_into_words=is_split_into_words)
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

def create_dataset_function(path, args):
    dataset_name = args.dataset_name
    cache_dir = ''
    offline = False
    transformer_name = f"{dataset_name}_{args.name_model}_{args.sample_length}"
    if args.low_resource:
        transformer_name += "low_resource"
    process_fn = None
    filter_fn = None

    if dataset_name == "semeval2014":
        def process_fn(rows):
            split = True
            pos_list = []
            pos_e_list = []
            label_list = []
            input_ids = []
            position_ids = []
            attention_masks = []
            for i in range(len(rows['text'])):
                text = rows['text'][i]
                terms = rows['aspectTerms'][i]
                for term in terms:
                    label = term['polarity']
                    if label == 'negative':
                        label = 0
                    elif label == 'positive':
                        label = 1
                    else:
                        label = 2
                    pack = _encode_single_text(text, args,  is_split_into_words=split)
                    start1, end1 = int(term['from']), int(term['to'])
                    pos1 = tokenizers[args.name_model+"-addprefix"](text[:start1], is_split_into_words=split)['input_ids'].__len__() - 2
                    pos1_e = tokenizers[args.name_model+"-addprefix"](text[:end1], is_split_into_words=split)['input_ids'].__len__() - 2
                    if pos1 == 0:
                        pos1 += 1
                    pos_list.append(pos1)
                    pos_e_list.append(pos1_e)
                    label_list.append(label)
                    input_ids.append(np.array(pack['input_ids'], dtype=np.int64))
                    position_ids.append(np.array(pack['position_ids'], dtype=np.int64))
                    attention_masks.append(np.array(pack['attention_mask'], dtype=np.int64))
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_masks,
                'label': label_list,
                'pos': pos_list,
                'pos_e': pos_e_list
            }
    elif dataset_name == "stsb":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence1'], row['sentence2'], args), float(row['label'])
            label = int(label * 1000)
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'stsb_label': label
            }
    elif dataset_name == "multirc":
        def process_fn(row):
            pack, label = _encode_double_text(row['question']+row['answer'], row['paragraph'] , truncation="only_second", args=args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "emotion":
        def process_fn(row):
            pack, label = _encode_single_text(row['text'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "sst2":
        def process_fn(row):
            pack, label = _encode_single_text(row['sentence'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "wic":
        def process_fn(row):
            split = True
            pack = _encode_double_text(row['sentence1'], row['sentence2'], args,  is_split_into_words=split)
            label = int(row['label'])
            start1, end1 = int(row['start1']), int(row['end1'])
            start2, end2 = int(row['start2']), int(row['end2'])
            pos1 = tokenizers[args.name_model+"-addprefix"](row['sentence1'][:start1], is_split_into_words=split)['input_ids'].__len__() - 2
            pos1_e = tokenizers[args.name_model+"-addprefix"](row['sentence1'][:end1], is_split_into_words=split)['input_ids'].__len__() - 2
            if pos1 == 0:
                pos1 += 1
            pos2 = tokenizers[args.name_model+"-addprefix"](row['sentence2'][:start2], is_split_into_words=split)['input_ids'].__len__() - 2
            pos2_e = tokenizers[args.name_model+"-addprefix"](row['sentence2'][:end2], is_split_into_words=split)['input_ids'].__len__() - 2
            if pos2 == 0:
                pos2 += 1
            pos2 += tokenizers[args.name_model+"-addprefix"](row['sentence1'], is_split_into_words=split)['input_ids'].__len__()
            pos2_e += tokenizers[args.name_model+"-addprefix"](row['sentence1'], is_split_into_words=split)['input_ids'].__len__()
            # new_sentence = (f"{row['sentence1']} {row['sentence2']} Does {row['word']} have the same meaning in both sentences?")
            # pack = _encode_single_text(new_sentence, args)
            # label = int(row['label'])

            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label,
                'pos1': pos1,
                'pos1_e': pos1_e,
                'pos2': pos2,
                'pos2_e':pos2_e,
            }
    elif dataset_name == 'cb':
        def process_fn(row):
            pack, label = _encode_double_text(row['premise'], row['hypothesis'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "rte":
        def process_fn(row):
            pack, label = _encode_double_text(row['premise'], row['hypothesis'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "boolq":
        def process_fn(row):
            pack, label = _encode_double_text(row['passage'], row['question'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "copa":
        def process_fn(row):
            type = row['question']
            premise, choice1, choice2 = row['premise'], row['choice1'], row['choice2']
            premise = premise[:-1]
            choice1 = choice1[0].lower() + choice1[1:]
            choice2 = choice2[0].lower() + choice2[1:]
            if type=='cause':
                sentence1 = premise + ' because ' + choice1
                sentence2 = premise + ' because ' + choice2
            else:
                sentence1 = premise + ' so ' + choice1
                sentence2 = premise + ' so ' + choice2
                pass
            pack_1 = _encode_single_text(sentence1, args)
            pack_2 = _encode_single_text(sentence2, args)
            label = int(row['label'])
            return {
                'input_ids_1': np.array(pack_1['input_ids'], dtype=np.int64),
                'input_ids_2': np.array(pack_2['input_ids'], dtype=np.int64),
                'position_ids_1': np.array(pack_1['position_ids'], dtype=np.int64),
                'position_ids_2': np.array(pack_2['position_ids'], dtype=np.int64),
                'attention_mask_1': np.array(pack_1['attention_mask'], dtype=np.int64),
                'attention_mask_2': np.array(pack_2['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "mrpc":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence1'], row['sentence2'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "qnli":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence'], row['question'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "qqp":
        def process_fn(row):
            pack, label = _encode_double_text(row['question1'], row['question2'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "cola":
        def process_fn(row):
            pack, label = _encode_single_text(row['sentence'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "wnli":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence1'], row['sentence2'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name in ['squad', 'squad_v2']:
        def process_fn(row):
            question = row['question'].strip()
            pack = _encode_double_text(question, row['context'], args, truncation='only_second')
            answers = row['answers']
            label = 0 if len(answers["text"]) == 0 else 1
            assert len(answers) <= 4

            start_list = []
            end_list = []
            for i in range(len(answers["text"])):
                texts = answers["text"][i]
                start = answers["answer_start"][i]

                start_pos = tokenizers[args.name_model](row['context'][:start])['input_ids'].__len__() - 2
                if start_pos == 0:
                    start_pos += 1
                end_pos = tokenizers[args.name_model](row['context'][:(start+len(texts))])['input_ids'].__len__() - 2

                start_pos += tokenizers[args.name_model](question)['input_ids'].__len__()
                end_pos += tokenizers[args.name_model](question)['input_ids'].__len__()
                if end_pos > 511:
                    print('exceed 512')
                    start_pos = 0
                    end_pos = 0
                # else:
                #     print(texts, tokenizers[args.name_model].decode(pack['input_ids'][start_pos:end_pos+1]))
                #     print(len(texts), len(tokenizers[args.name_model].decode(pack['input_ids'][start_pos:end_pos+1])))
                start_list.append(start_pos)
                end_list.append(end_pos)

            assert len(start_list)<=7
            while len(start_list) < 7:
                start_list.append(-1)
                end_list.append(-1)
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label,
                'start_list': np.array(start_list, dtype=np.int64),
                'end_list': np.array(end_list, dtype=np.int64)
            }
    elif dataset_name in ['conll2003']:
        def process_fn(row):
            labels = row['ner_tags']
            pack = _encode_single_text(row['tokens'], args, is_split_into_words=True)
            word_ids = [None]
            for j, word in enumerate(row['tokens']):
                token = tokenizers[args.name_model].encode([word], add_special_tokens=False, is_split_into_words=True)
                word_ids += [j] * len(token)
            word_ids += [None]
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                    # label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            label_ids = label_ids + ([-100] * (len(pack['input_ids']) - len(label_ids)))
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': np.array(label_ids, dtype=np.int64),
            }
    elif dataset_name in ['wsc']:
        import en_core_web_sm
        def process_fn(row):
            nlp = en_core_web_sm.load()
            pad = lambda a: a[0:args.sample_length] if len(a) > args.sample_length else a + [1] * (args.sample_length-len(a))
            text = row['text']
            query = row['span1_text']
            if query[-1]=='.' or query[-1] == ',':
                query = query[:-1]
            sentence = nlp(text)
            tokenizer = tokenizers[args.name_model+'-addprefix']
            encoded_text = tokenizer(text, is_split_into_words=True)['input_ids']
            start2 = row['text'].find(row['span2_text'])
            pron_index = tokenizer(row['text'][:start2], is_split_into_words=True)['input_ids'].__len__() - 2
            encoded_query = tokenizer(query, is_split_into_words=True)['input_ids'][1:-1]
            prefix = encoded_text[:pron_index]
            suffix = encoded_text[pron_index+1 :]
            query_tokens = pad(prefix + encoded_query + suffix)
            query_tokens = np.array(query_tokens)
            query_mask = np.zeros_like(query_tokens)
            query_mask[len(prefix):len(prefix)+len(encoded_query)] = 1
            # print(tokenizer.decode(encoded_query))
            # print(tokenizer.decode(query_tokens)[:500])
            # breakpoint()
            #extend chunks
            noun_chunks = {(nc.start, nc.end) for nc in sentence.noun_chunks}
            np_start, cur_np = 0, "NONE"
            for i, token in enumerate(sentence):
                np_type = token.pos_ if token.pos_ in {"NOUN", "PROPN"} else "NONE"
                if np_type != cur_np:
                    if cur_np != "NONE":
                        noun_chunks.add((np_start, i))
                    if np_type != "NONE":
                        np_start = i
                    cur_np = np_type
            if cur_np != "NONE":
                noun_chunks.add((np_start, len(sentence)))
            noun_chunks = [sentence[s:e] for (s, e) in sorted(noun_chunks)]

            #exclude pron and query
            noun_chunks = [
                nc
                for nc in noun_chunks
                if (nc.lemma_ != "-PRON-" and not all(tok.pos_ == "PRON" for tok in nc))
            ]
            excl_txt = [query.lower()]
            filtered_chunks = []
            for chunk in noun_chunks:
                lower_chunk = chunk.text.lower()
                found = False
                for excl in excl_txt:
                    if (lower_chunk in excl or excl in lower_chunk) or lower_chunk == excl:
                        found = True
                        break
                if not found:
                    filtered_chunks.append(chunk)

            noun_chunks = filtered_chunks

            #calc cand_tokens
            cand_token_list = []
            cand_mask_list = []
            for cand_span in noun_chunks:
                encoded_cand = tokenizer(cand_span.text, is_split_into_words=True)['input_ids'][1:-1]
                cand_tokens = pad(prefix + encoded_cand + suffix)
                cand_tokens = np.array(cand_tokens)
                cand_mask = np.zeros_like(cand_tokens)
                cand_mask[len(prefix):len(prefix)+len(encoded_cand)] = 1
                cand_token_list.append(cand_tokens)
                cand_mask_list.append(cand_mask)

            if len(cand_token_list) > args.max_cand_len:
                cand_token_list = cand_token_list[:args.max_cand_len]
                cand_mask_list = cand_mask_list[:args.max_cand_len]
            for i in range(args.max_cand_len - len(cand_token_list)):
                cand_token_list.append([-1] * args.sample_length)
                cand_mask_list.append([0] * args.sample_length)
            cand_tokens = np.stack(cand_token_list)
            cand_mask = np.stack(cand_mask_list)
            cand_tokens = cand_tokens.reshape([-1])
            cand_mask = cand_mask.reshape([-1])

            label = int(row['label'])
            return {
                'label': label,
                'query_tokens': query_tokens,
                'query_mask' : query_mask,
                'cand_tokens': cand_tokens,
                'cand_mask': cand_mask
            }
        def filter_fn(row):
            if "tokens" in row.keys() and len(row['tokens']) == 0:
                return False
            else:
                return True
    else:
        raise Exception('Dataset name is wrong.')
    return load_hf_dataset(path, process_fn, filter_fn = filter_fn, columns = get_dataset_keys(dataset_name), cache_dir=cache_dir, offline=offline, transformer_name=transformer_name, low_resource=args.low_resource)


def get_loss_metrics(logits, labels, dataset_name, **extra_data):
    if dataset_name in ['rte', 'boolq', 'wic', 'mrpc', 'qnli', 'qqp', 'cola', 'wnli', 'sst2', 'multirc']:
        pred = logits.contiguous().float().squeeze(-1)
        # if dataset_name == 'multirc':
        #     pos_weight = torch.ones([1], dtype=pred.dtype, device=pred.device) * 1.25
        # else:
        #     pos_weight = None
        pos_weight = None
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            labels.float(),
            pos_weight = pos_weight
        )
        true_pos = ((pred > 0.).long() * labels).sum() * 1.0
        false_pos = ((1-(pred > 0.).long()) * labels).sum() * 1.0
        true_neg = ((1-(pred > 0.).long()) * (1-labels)).sum() * 1.0
        false_neg = ((pred > 0.).long() * (1-labels)).sum() * 1.0
        acc = ((pred > 0.).long() == labels).sum() / labels.numel()
        eval_acc = ((pred > 0.).long() == labels).float()

        return loss, {'acc': acc, 'tp': true_pos, 'fp': false_pos, 'tn': true_neg, 'fn': false_neg, 'eval_acc': eval_acc}
    elif dataset_name in ['stsb']:
        pred = logits.contiguous().float().squeeze(-1)
        labels = labels.float()/1000
        loss = torch.nn.functional.mse_loss(
            pred,
            labels,
        )

        vpred = pred - torch.mean(pred)
        vlabels = labels - torch.mean(labels)

        pearson = torch.sum(vpred * vlabels) / (torch.sqrt(torch.sum(vpred ** 2)) * torch.sqrt(torch.sum(vlabels ** 2)))

        return loss, {'pearson': pearson, 'eval_stsb_pred': pred, 'eval_stsb_labels': labels}
    elif dataset_name=="copa":
        bz = logits.shape[0] // 2
        logits = logits.squeeze(-1).reshape(2, bz).permute(1, 0)
        pred = logits.contiguous().float()
        loss = torch.nn.functional.cross_entropy(
            pred,
            labels
        )
        acc = (torch.argmax(pred, dim=1).long() == labels).sum() / labels.numel()
        eval_acc = (torch.argmax(pred, dim=1).long() == labels).float()
        return loss, {'acc': acc, 'eval_acc':eval_acc}
    elif dataset_name in ['cb', 'emotion', 'semeval2014']:
        pred = logits.contiguous().float()
        loss = torch.nn.functional.cross_entropy(pred, labels)
        acc = (torch.argmax(pred, dim=-1).long() == labels).sum() / labels.numel()
        eval_acc = (torch.argmax(pred, dim=-1).long() == labels).float()
        return loss, {'acc': acc, 'eval_acc':eval_acc}
    elif dataset_name in ['conll2003']:
        pred = logits.contiguous().float().squeeze(-1)
        loss = torch.nn.functional.cross_entropy(
            pred.view(pred.shape[0] * pred.shape[1], pred.shape[2]),
            labels.view(labels.shape[0] * labels.shape[1])
        )
        pred = pred.argmax(dim=2).long()
        acc = (pred == labels).sum()/ (labels.numel() - (labels == -100).sum())
        return loss, {'acc': acc, 'eval_pred':pred, 'eval_labels':labels}
    elif dataset_name in ['squad', 'squad_v2']:
        # For negative examples, abstaining receives a score of 1,
        # and any other response gets 0, for both exact match and F1.
        start_logits, end_logits, cls_logits = logits
        start_logits = start_logits.contiguous().float()
        end_logits = end_logits.contiguous().float()
        cls_logits = cls_logits.contiguous().float()
        start_list = extra_data['start_list']
        end_list = extra_data['end_list']
        pred = cls_logits.contiguous().float().squeeze(-1)
        pred = torch.nn.functional.sigmoid(pred)
        loss1 = torch.nn.functional.binary_cross_entropy(
            pred,
            labels.float()
        )
        acc = ((pred > 0.).long() == labels).sum() / labels.numel()
        train_start_list = start_list[:,:1].squeeze(-1)
        train_end_list = end_list[:, :1].squeeze(-1)

        loss2 = torch.zeros_like(loss1, dtype=loss1.dtype, device=loss1.device)
        cnt = 0
        for i in range(start_list.shape[0]):
            if train_start_list[i] != -1:
                cnt += 1
                loss2 = loss2 + torch.nn.functional.cross_entropy(
                    start_logits[i],
                    train_start_list[i]
                )
                loss2 = loss2 + torch.nn.functional.cross_entropy(
                    end_logits[i],
                    train_end_list[i]
                )
        if cnt != 0:
            loss2 = loss2 / cnt

        em = []
        f1 = []
        start_pred = start_logits.argmax(dim=1)
        end_pred = []
        for i in range(start_pred.shape[0]):
            end_pred.append(end_logits[i][start_pred[i]:].argmax(dim=0) + start_pred[i])

        for i in range(start_list.shape[0]):
            now_em = 0
            max_f1 = 0
            if pred[i] > 0.5:
                for j in range(7):
                    start_index, end_index = start_list[i][j], end_list[i][j]
                    if start_index == -1:
                        break
                    if start_pred[i]==start_index and end_pred[i]==end_index:
                        now_em = 1
                    hit = max(0, min(end_pred[i], end_index) - max(start_index, start_pred[i]) + 1)
                    precision = hit/(end_pred[i]-start_pred[i]+1)
                    recall = hit/(end_index-start_index+1)
                    max_f1 = max(max_f1, 2*(precision*recall)/(precision+recall))

            em.append(now_em)
            f1.append(max_f1)
        eval_em = torch.tensor(em, dtype=cls_logits.dtype, device=cls_logits.device)
        eval_f1 = torch.tensor(f1, dtype=cls_logits.dtype, device=cls_logits.device)
        em = sum(eval_em.split(1,0))[0]/eval_em.numel()
        f1 = sum(eval_f1.split(1,0))[0]/eval_f1.numel()
        if loss2 is not None:
            loss = loss1 + loss2
        else:
            loss = loss1
        return loss, {'acc':acc, 'loss1':loss1.data, 'em':em, 'f1':f1, 'eval_em': eval_em, 'eval_f1':eval_f1}

from datasets import load_metric
metric_max = {'acc':torch.tensor([0]), 'f1':torch.tensor([0]), 'mc':torch.tensor([0]), 'pearson':torch.tensor([0]), 'spearmanr':torch.tensor([0])}
def handle_metrics(metrics):
    if 'eval_stsb_labels' in metrics.keys():
        pred = metrics['eval_stsb_pred']
        labels = metrics['eval_stsb_labels']
        metric = load_metric('glue', 'stsb')
        results = metric.compute(predictions=pred, references=labels)
        metric_max['pearson'] = torch.max(metric_max['pearson'], torch.tensor(results["pearson"]))
        metric_max['spearmanr'] = torch.max(metric_max['spearmanr'], torch.tensor(results["spearmanr"]))
        return {"pearson": torch.tensor(results["pearson"]), "spearmanr": torch.tensor(results["spearmanr"]), "max_pear": metric_max['pearson'], "max_spear":metric_max['spearmanr']}
    elif 'eval_pred' in metrics.keys():
        pred = metrics['eval_pred']
        labels = metrics['eval_labels']
        true_predictions = []
        true_labels = []
        label_list = ['O', 'B-PER', "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        for i in range(pred.shape[0]):
            prediction = []
            label = []
            for j in range(pred.shape[1]):
                if labels[i,j] != -100:
                    prediction.append(label_list[pred[i,j]])
                    label.append(label_list[labels[i,j]])
            true_predictions.append(prediction)
            true_labels.append(label)
        metric = load_metric('seqeval')
        results = metric.compute(predictions=true_predictions, references=true_labels)
        metric_max['f1'] = torch.max(metric_max['f1'], torch.tensor(results["overall_f1"]))
        return {"f1": torch.tensor(results["overall_f1"]), 'max_f1': metric_max['f1']}
    elif 'em' in metrics.keys():
        eval_em = sum(metrics['eval_em'].split(1,0))/len(metrics['eval_em'])
        eval_f1 = sum(metrics['eval_f1'].split(1,0))/len(metrics['eval_f1'])
        return {'em': eval_em, 'f1': eval_f1}
    elif 'tp' in metrics.keys():
        acc = sum(metrics['eval_acc'].split(1,0))/len(metrics['eval_acc'])
        TP = sum(metrics['tp'].split(1,0))
        TN = sum(metrics['tn'].split(1,0))
        FP = sum(metrics['fp'].split(1,0))
        FN = sum(metrics['fn'].split(1,0))
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*(Precision*Recall)/(Precision+Recall)
        MC = (TP*TN-FP*FN)/torch.sqrt((TP+FP)*(FN+TP)*(FN+TN)*(FP+TN))
        metric_max['acc'] = torch.max(metric_max['acc'], acc)
        metric_max['f1'] = torch.max(metric_max['f1'], F1)
        metric_max['mc'] = torch.max(metric_max['mc'], MC)
        # print(metric_max)
        # input()
        return {'acc': acc, 'f1': F1, 'mc':MC, 'acc_max': metric_max['acc'], 'f1_max': metric_max['f1'], 'mc_max': metric_max['mc']}
    else:
        acc = sum(metrics['eval_acc'].split(1,0))/len(metrics['eval_acc'])
        metric_max['acc'] = torch.max(metric_max['acc'], acc)
        return {'acc': acc, 'acc_max': metric_max['acc']}

class ChildTuningAdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            reserve_p = 1.0,
            mode = None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================
                if self.mode is not None and self.gradient_mask is not None:
                    if self.mode == 'ChildTuning-D':
                        if p in self.gradient_mask:
                            grad *= self.gradient_mask[p]
                    else:
                        # ChildTuning-F
                        grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                        grad *= grad_mask.sample() / self.reserve_p
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

def calc_mask(model, args, train_data, forward_step):
    timers = Timers()
    N = len(train_data)//args.epochs
    # if N > 200:
    #     N = 200
    print(f"{N} samples to calc mask")

    model.train()
    gradient_mask = dict()
    for name, params in model.named_parameters():
        if 'transformer.layers' in name:
            gradient_mask[params] = params.new_zeros(params.size())
    for _ in tqdm(range(N)):
        loss, _ = forward_step(train_data, model, args, timers)
        loss.backward()

        for name, params in model.named_parameters():
            if 'transformer.layers' in name:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                gradient_mask[params] += (params.grad ** 2)
        model.zero_grad()
    print('Calculate Fisher Information')

    # Numpy
    r = None
    for k, v in gradient_mask.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    polar = np.percentile(r, (1-args.reserve_p)*100)
    for k in gradient_mask:
        gradient_mask[k] = gradient_mask[k] >= polar

    print('Polar => {}'.format(polar))

    # for name, params in model.named_parameters():
    #     if 'transformer.layers' in name:
    #         cnt = gradient_mask[params].sum()
    #         sz = gradient_mask[params].size()
    #         cnt2 = 1
    #         for szz in sz:
    #             cnt2 *= szz
    #         print(name[18:], f"{cnt}/{cnt2}", (cnt/cnt2).cpu().numpy().tolist())
    return gradient_mask

def set_optimizer_mask(model, args, train_data, optimizer, forward_step):
    train_data = iter(train_data)
    if args.child_type == "ChildTuning-D":
        grad_mask = calc_mask(model, args, train_data, forward_step)
        optimizer.optimizer.set_gradient_mask(grad_mask)