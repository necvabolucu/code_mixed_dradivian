import json
import torch
from argparse import Namespace
import numpy as np
import random
import pickle
import sys, os, time
import argparse
import re, csv
from transformers import BertTokenizer
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
import copy

global spacy_parser, tokenizer

# data class
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
class DataProcessor_a(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. 
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
class MixProcessor(DataProcessor_a):
    """Processor for the Mix data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

        
class InputRawExample(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label

class Graph(object):
    """docstring for Graph"""
    def __init__(self, n):
        super(Graph, self).__init__()
        self.n = n
        self.link_list = []
        self.vis = [0] * self.n
        for i in range(self.n):
            self.link_list.append([])

    def add_edge(self, u, v):
        if u == v:
            return
        self.link_list[u].append(v)
        self.link_list[v].append(u)

    def bfs(self, start, dist):
        que = [start]
        self.vis[start] = 1
        for _ in range(dist):
            que2 = []
            for u in que:
                #self.vis[u] = 1
                for v in self.link_list[u]:
                    if self.vis[v]:
                        continue
                    que2.append(v)
                    self.vis[v] = 1
            que = copy.deepcopy(que2)
            

    def solve(self, start, dist):
        self.vis = [0] * self.n
        self.bfs(start, dist)
        self.vis[0] = 1
        return copy.deepcopy(self.vis)
    

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns


def get_config(config_filepath):
  with open(config_filepath, "r") as config_file:
    conf = json.load(config_file, object_hook=lambda d: Namespace(**d))
  return conf

def get_device():
    "Get GPU if available"

    use_gpu = torch.cuda.is_available()
    return torch.device("cuda" if use_gpu else "cpu"), use_gpu

def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if True:
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    


def create_dataset(input_file):
    dataset = pickle.load(open(input_file, 'rb'))
    return dataset


def read_file(path):
  f = open(path, "r")

  text = []
  label = []
  for line in f.readlines():
    line = line.strip("\n").strip(" ").split("\t")
    
    if line[-1] == "":
      text.append(" ".join(line[:-2]))
      label.append(line[-2])
    else:
      text.append(" ".join(line[:-1]))
      label.append(line[-1])
  return text, label
task_processors = {
    'mix' : MixProcessor,
}
s2l = {}
def text2label(label): 
  label_encoded = []
  for item in label:
    if item not in s2l.keys():
      s2l[item] = len(s2l)
    label_encoded.append(s2l[item])
  return label_encoded


def process(text, label, task):
    lang = "multi"
    dist = 3
    # text: str
    # label: list[str] or int
    global spacy_parser, tokenizer
    if lang == 'multi':
        local_tokenizer = tokenizer['multilingual-cased']# if do_lower_case else tokenizer['en-cased']

        while '  ' in text:
            text = text.replace('  ', ' ')
        doc = spacy_parser(text)

        tokens = ['[CLS]']
        for token in doc:
            token._.tid = len(tokens)
            tokens.append(token.text)
        
        G = Graph(len(tokens))
        for token in doc:
            if token.dep_ == 'ROOT':
                continue
            G.add_edge(token._.tid, token.head._.tid)


        ntokens = []
        ws = []

        for i, token in enumerate(tokens):
            if token == '[CLS]':
                ntokens.append(token)
                ws.append(1)
            else:
                sub_tokens = local_tokenizer.tokenize(token)
                ws.append(len(sub_tokens))
                for j, st in enumerate(sub_tokens):
                    ntokens.append(st)

        dep_att_mask = []
        for i, token in enumerate(tokens):
            vis = G.solve(i, dist)
            
            if i-1>=0:
                vis_tmp = G.solve(i-1, dist)
                for j in range(len(vis_tmp)):
                    vis[j] |= vis_tmp[j]

            if i+1<len(tokens):
                vis_tmp = G.solve(i+1, dist)
                for j in range(len(vis_tmp)):
                    vis[j] |= vis_tmp[j]
            
            mask = []
            for j in range(len(vis)):
                for k in range(ws[j]):
                    mask.append(vis[j])

            assert len(ntokens) == len(mask), ntokens
            mask.append(1)

            for k in range(ws[i]):
                dep_att_mask.append(mask)

        ntokens.append('[SEP]')
        if isinstance(label, list):
            labels = []
            loss_mask = []
            for i, token in enumerate(tokens):
                if token == '[CLS]':
                    labels.append('O')
                    loss_mask.append(0)
                else:
                    for j in range(ws[i]):
                        labels.append(label[i-1])
                        loss_mask.append(1 if j==0 else 0)
            labels.append('O')
            loss_mask.append(0)
            label_s2i = {}
            for i, s in enumerate(task_processors[task]().get_labels()):
                label_s2i[s] = i
            labels = [label_s2i[s] for s in labels]
        else:
            labels = [label]

        dep_att_mask.append([1] * len(ntokens))
        for j in range(len(dep_att_mask[0])):
            dep_att_mask[0][j] = 1

        quote_idx = []
        for i, w in enumerate(ntokens):
            w = w.lower()
            if w in '!\"#$%&\'()*+,.-/:;<=>?@[\\]^_':
                quote_idx.append(i)

        for i in range(len(ntokens)):
            for j in quote_idx:
                dep_att_mask[i][j] = 1
        
        input_ids = local_tokenizer.convert_tokens_to_ids(ntokens)
    else:
        raise print("error")

    example = {
        'input_ids': input_ids,
        'dep_att_mask': dep_att_mask,
        'labels': labels,
    }
    if isinstance(label, list):
        example['loss_mask'] = loss_mask
    return example


def prepare_dataset(cfg):
  task = "classification"
  train_file_name = cfg.train_file
  dev_file_name = cfg.dev_file
  test_file_name = cfg.test_file
  raw_examples = []
  lines, labels = read_file(train_file_name)
  label_encoded = text2label(labels)
  for line, label in zip(lines,label_encoded):
      raw_examples.append(InputRawExample(line, int(label)))
  examples = []
  for item in tqdm(raw_examples, desc='Convert'):
      examples.append(process(item.text, item.label,task))

  pickle.dump(examples, open('dataset/%s.%s.pkl'%(task, cfg.save_syntax_train_file), 'wb'))


  raw_examples = []
  lines, labels = read_file(dev_file_name)
  label_encoded = text2label(labels)
  for line, label in zip(lines,label_encoded):
      raw_examples.append(InputRawExample(line, int(label)))
  examples = []
  for item in tqdm(raw_examples, desc='Convert'):
      examples.append(process(item.text, item.label,task))

  pickle.dump(examples, open('dataset/%s.%s.pkl'%(task, cfg.save_syntax_dev_file), 'wb'))

  raw_examples = []
  lines, labels = read_file(test_file_name)
  label_encoded = text2label(labels)
  for line, label in zip(lines,label_encoded):
      raw_examples.append(InputRawExample(line, int(label)))
  examples = []
  for item in tqdm(raw_examples, desc='Convert'):
      examples.append(process(item.text, item.label,task))

  pickle.dump(examples, open('dataset/%s.%s.pkl'%(task, cfg.save_syntax_test_file), 'wb'))


def compute_metrics(pred): 
    labels = pred.label_ids 
    preds = pred.predictions.argmax(-1) 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted') 
    calculate_scores(labels, preds)
    acc = accuracy_score(labels, preds) 
    return { 
        'Accuracy': acc, 
        'F1': f1, 
        'Precision': precision, 
        'Recall': recall 
    }

def calculate_scores(label_list, prediction_list):
    print(classification_report(label_list, prediction_list))
    print(precision_score(label_list, prediction_list, average="weighted"), recall_score(label_list, prediction_list, average="weighted"), f1_score(label_list, prediction_list, average="weighted"))
    cf_matrix = confusion_matrix(label_list, prediction_list)
    sns.heatmap(cf_matrix)