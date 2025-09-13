import torch
from torch.utils.data import Dataset


class VerilogDataset(Dataset):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        inputs = self.tokenizer(self.data_list[index], return_tensors="pt", padding=True).to('cuda')
        attention_mask = inputs['attention_mask'][0].to('cuda')
        input_ids = inputs['input_ids'][0].to('cuda')

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return inputs

    def __len__(self):
        return len(self.data_list)
