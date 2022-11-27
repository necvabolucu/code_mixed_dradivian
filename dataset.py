import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
      selected_text = self.text[idx]
      selected_label = torch.tensor(int(self.label[idx]), dtype=torch.long)
      inputs = self.tokenizer.encode_plus(
          selected_text ,
          None,
          add_special_tokens=True,
          max_length=self.max_len,
          padding='max_length',
          return_token_type_ids=True,
          truncation=True,
          return_attention_mask=True,
      )
      
      tokens = torch.tensor(inputs["input_ids"], dtype=torch.long)
      token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
      attn_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

      return tokens, token_type_ids, attn_mask, selected_label