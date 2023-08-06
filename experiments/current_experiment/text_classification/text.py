import torch
import torch.nn as nn
from torch.utils.data import Dataset 
from transformers import AutoConfig, AutoTokenizer, AutoModel
import typing 
import logging 
import os
import pandas 
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(
    filename=os.path.join("../experiment_logs/text_classification.log")
)
logger.addHandler(file_handler)
logger.setLevel(level=logging.CRITICAL)

class EncodeError(BaseException):
    def __init__(self, msg):
        self.msg = msg 

class TextClassificationDataset(Dataset):
    
    def __init__(self,
        texts: typing.List[str],
        label_dict: typing.Dict[str, int],
        labels: typing.List[str],
        model_name: str = "distilbert-base-uncased",
        max_sequence_length: int = 512,):
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
        )
        self.texts = texts
        self.max_sequence_length = max_sequence_length

        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]
        
        if labels is not None and label_dict is None:
            self.label_dict = dict(zip(sorted(set(labels)))), range(len(set(labels)))
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> typing.Mapping[str, torch.Tensor]:

        text = self.texts[idx]
        x_encoded = self.tokenizer.encode(
            text=text,
            add_special_tokens=True,
            return_tensors='pt'
        ).squeeze(0)
    
        true_size = x_encoded.size(0)
        pad_size = self.max_sequence_length - true_size 
        pad_ids = torch.Tensor([self.pad_vid] * pad_size, ).long()
        x_tensor = torch.cat(tensors=(x_encoded, pad_ids))
        
        mask = torch.ones_like(x_encoded, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        mask = torch.cat((mask, mask_pad))

        output_dict = {
            "features": x_tensor,
            'attention_mask': mask
        }

        if self.labels is not None:
            y = self.labels[idx]
            y_encoded = torch.Tensor(
                [self.label_dict.get(y, -1)]
            ).long().squeeze(0)
            output_dict["targets"] = y_encoded

        return output_dict


class BertTextClassifier(nn.Module):
    
    def __init__(self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = None
        ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name)

        self.bert = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        # initializing classifier linear layers
        self.pre_classifier = torch.Linear(self.config.dim, self.config.dim)
        self.classifier = torch.Linear(self.config.dim, num_classes)
        self.dropout = torch.Dropout(self.config.seq_classif_dropout)

    def forward(self, features, attention_mask=None, head_mask=None):

        if attention_mask is None:
            raise AssertionError("Attension Mask is None")

        distilbert_output = self.distilbert(
            input_ids=features,
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        hidden_state = distilbert_output[0]  
        pooled_output = hidden_state[:, 0] 
        pooled_output = self.pre_classifier(pooled_output)  
        pooled_output = torch.ReLU()(pooled_output)  
        pooled_output = self.dropout(pooled_output)  
        logits = self.classifier(pooled_output)
        return logits

class TFIDFVectorizedDataset(dict):

    def __init__(self, text_data: pandas.DataFrame):
        self.text_data = text_data 
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def get_vectorized_df(self):
        try:
            encoded = self.vectorizer.fit_transform(
                raw_documents=self.text_data,
                lowercase=True
            )
            return encoded.toarray() 
        except Exception as encode_err:
            logger.debug(encode_err) 
            raise EncodeError(msg=encode_err.args)
