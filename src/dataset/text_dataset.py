import torch
import logging 
import typing
from transformers import AutoTokenizer 
from torch.utils.data import Dataset

logger = logging.getLogger("transformers.tokenization_utils")
logger.setLevel(logging.FATAL)

class TextClassificationDataset(Dataset):

    def __init__(self,
                texts: typing.List[str],
                labels: typing.List[str] = None,
                label_dict: typing.Dict[str, int] = None,
                max_seq_length: int = 512,
                model_name: str = "distilbert-base-uncased"):
        
        self.texts = texts
        self.labels = labels 
        self.label_dict = label_dict 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length

        # setting up vids 
        self.class_vid = self.tokenizer.vocab["[CLS]"]
        self.set_vid = self.tokenizer.vocab["[SEP]"]
        self.pad_vid = self.tokenizer.vocab['[PAD]']
    
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, key):
        text_data = self.texts[key]
        pass



