import utils
import torch
import os
from dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score


class TextModel(torch.nn.Module):
  def __init__(self, embedding, out_n=6, dropout=0.1):
    super(TextModel,self).__init__()
    self.embedding = embedding
    self.dropout = torch.nn.Dropout(dropout)
    self.decoder = torch.nn.Linear(768, out_n)
 
  def forward(self, input_ids, attention_mask, token_type_ids=None):
    output = self.embedding(input_ids, attention_mask) 
    output = self.dropout(output[0][:,0,:])
    output = self.decoder(output)
    return output

def save_checkpoint(state, location, name):
  """save the models
  input:
  state : dict (the parameters of the model will be saved)
  file_path : string (the path wehere the model will be saved)
  """
  filepath = os.path.join(location, name)
  torch.save(state, filepath)

def load_checkpoint(location):
  """save the models
  input:
  file_path : string (the path where the model will be saved)
  output:
  model : torch nn.Module (the loaded model)
  """
  model = torch.load(location, map_location=torch.device('cpu'))
  return model


def train(model, optimizer, device, criterion, train_dl):
  model.train()
  for batch in train_dl:
    optimizer.zero_grad()
    tokens, token_type_ids, attn_mask, label = batch
    tokens, token_type_ids, attn_mask, label = tokens.to(device), token_type_ids.to(device), attn_mask.to(device), label.to(device)
    output = model(tokens, attn_mask, token_type_ids)

    loss = criterion(output, label)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    optimizer.step()



def evaluate(model, device, criterion, dl):
  total_loss = 0.
  prediction_list = []
  label_list = []

  with torch.no_grad():
    model.eval()
    for batch in dl:
      tokens, token_type_ids, attn_mask, label = batch
      tokens, token_type_ids, attn_mask, label = tokens.to(device), token_type_ids.to(device), attn_mask.to(device), label.to(device)
      output = model(tokens, attn_mask, token_type_ids)
          
      loss = criterion(output, label)
      prediction_list.extend(torch.argmax(output, dim=1).cpu().detach().numpy())

      label_list.extend(label.data.cpu().detach().numpy())

      total_loss += loss.item() 
    #prediction_list = (np.array(prediction_list) >= 0.5).astype(int)
    return accuracy_score(label_list, prediction_list), total_loss/len(dl), label_list, prediction_list

def train_and_evaluate(cfg, model, optimizer, criterion, scheduler, device, train_dl, val_dl, test_dl=None):
    patience = 0
    best_val_acc = -999.9    
    
    for epoch in range(1, cfg.epoch_size+1):
        train(model, optimizer, device, criterion, train_dl)
        train_acc, train_loss, label_list, prediction_list = evaluate(model, device, criterion, train_dl)
        val_acc, val_loss, label_list, prediction_list = evaluate(model, device, criterion, val_dl)
        test_acc, test_loss, label_list, prediction_list = evaluate(model, device, criterion, test_dl)
        print("epoch %d , train acc is %f, train loss %f, the val acc is %f, val loss %f , the test acc is %f, test loss %f" % (epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
    
        if val_acc > best_val_acc:
          patience = 0
    
          save_checkpoint({'epoch': epoch , 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}, location='result/', name=cfg.save_path)
          best_val_acc = val_acc
          print("save the model...")
          print("epoch %d , the current best f is %f" % (epoch, best_val_acc))
    
        else:
            patience += 1
    
        if patience > cfg.patience_all:
            break
        scheduler.step()
      
def train_bert(cfg):
    train_text, train_label = utils.read_file(cfg.train_file)
    dev_text, dev_label = utils.read_file(cfg.dev_file)
    test_text, test_label = utils.read_file(cfg.test_file)
    
    train_label_encoded = utils.text2label(train_label)
    dev_label_encoded = utils.text2label(dev_label)
    test_label_encoded = utils.text2label(test_label)
    
    utils.set_seed(cfg)
    device = utils.get_device()
    
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)
    embedding = BertModel.from_pretrained(cfg.bert_model,output_hidden_states = True).to(device)
    model = TextModel(embedding, out_n=len(set(train_label)), dropout=cfg.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.init_weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    dl_train = DataLoader(CustomDataset(train_text, train_label_encoded, tokenizer,  cfg.max_length), num_workers=cfg.num_workers,  shuffle=True, batch_size=cfg.batch_size)
    dl_val= DataLoader(CustomDataset(dev_text, dev_label_encoded, tokenizer,  cfg.max_length), num_workers=cfg.num_workers,  shuffle=True, batch_size=cfg.batch_size)
    dl_test = DataLoader(CustomDataset(test_text, test_label_encoded, tokenizer,  cfg.max_length), num_workers=cfg.num_workers, shuffle=False, batch_size=cfg.batch_size)
    
    train_and_evaluate(cfg, model, optimizer, criterion, scheduler, device, dl_train, dl_val, dl_test)
    
    filepath = os.path.join('result',cfg.save_path)
    state_dict = load_checkpoint(filepath)
    model.load_state_dict(state_dict["state_dict"])
    
    test_acc, test_loss, label_list, prediction_list = evaluate(model, device, criterion, dl_test)
    calculate_scores(label_list, prediction_list)
    