from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import sys
import gzip
from datetime import datetime
from transformers import logging
import utils
import torch
from transformers import BertForSequenceClassification
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
from transformers import BertTokenizerFast
from transformers import TrainingArguments, Trainer
    
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


def cont_BERT(cfg):
    device = utils.get_device()
    model = AutoModelForMaskedLM.from_pretrained(cfg.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model)
 
    train_text, train_label = utils.read_file(cfg.train_file)
    dev_text, dev_label = utils.read_file(cfg.dev_file)
    test_text, test_label = utils.read_file(cfg.test_file)
    
    train_label_encoded = utils.text2label(train_label)
    dev_label_encoded = utils.text2label(dev_label)
    test_label_encoded = utils.text2label(test_label)
    
    train_dataset = TokenizedSentencesDataset(train_text, tokenizer, cfg.max_length)
    dev_dataset = TokenizedSentencesDataset(dev_text, tokenizer, cfg.max_length, cache_tokenization=True) 
    
    if cfg.do_whole_word_mask:
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mlm_prob)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mlm_prob)
        

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.epoch_size,
        evaluation_strategy="steps" if dev_dataset is not None else "no",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        eval_steps=cfg.save_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.save_steps,
        save_total_limit=1,
        prediction_loss_only=True,
        fp16=cfg.use_fp16
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )
    
    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    model.save_pretrained(cfg.output_dir)
    
    tokenizer = BertTokenizerFast.from_pretrained(cfg.output_dir, max_length=cfg.max_length)

    model = BertForSequenceClassification.from_pretrained(cfg.output_dir, num_labels=len(set(train_label_encoded)))
    model.to(device)
    
    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    val_encodings  = tokenizer(dev_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    
    train_dataset = MyDataset(train_encodings, train_label_encoded)
    val_dataset = MyDataset(val_encodings, dev_label_encoded)
    test_dataset = MyDataset(test_encodings, test_label_encoded)
    
    training_args = TrainingArguments(
        output_dir=cfg.output_dir_cont, 
        do_train=True,
        do_eval=True,
        #  The number of epochs, defaults to 3.0 
        num_train_epochs=cfg.epoch_size,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=32,
        # Number of steps used for a linear warmup
        warmup_steps=cfg.warmup_steps,                
        weight_decay=cfg.init_weight_decay,
        logging_strategy='steps',
       # TensorBoard log directory                 
        logging_dir='multi-class-logs',            
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps", 
        fp16=cfg.use_fp16,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        # the pre-trained model that will be fine-tuned 
        model=model,
         # training arguments that we defined above                        
        args=training_args,                 
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,            
        compute_metrics= utils.compute_metrics
        )
    
    trainer.evaluate(eval_dataset=test_dataset)