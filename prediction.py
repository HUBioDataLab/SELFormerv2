import os

os.environ["TOKENIZER_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset


from transformers import BertPreTrainedModel, RobertaConfig, RobertaTokenizerFast, Trainer

from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="bbbp", help="dataset selection.")
parser.add_argument("--tokenizer_name", default="data/RobertaFastTokenizer", metavar="/path/to/dataset/", help="Tokenizer selection.")
parser.add_argument("--pred_set", default="data/finetuning_datasets/classification/bbbp_mock/bbbp_mock_modelO_embeddings.pkl", metavar="/path/to/dataset/", help="Test set for predictions.")
parser.add_argument("--training_args", default= "data/finetuned_models/bbbp/training_args.bin", metavar="/path/to/dataset/", help="Trained model arguments.")
parser.add_argument("--model_name", default="data/finetuned_models/bbbp",  metavar="/path/to/dataset/", help="Path to the model.")
args = parser.parse_args()



# Model

class CustomClassificationHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(CustomClassificationHead, self).__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(input_dim, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MultiModalTransformers_For_Classification(BertPreTrainedModel):
    def __init__(self, config):
        super(MultiModalTransformers_For_Classification, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        combined_hidden_size = 2112
        self.classifier = CustomClassificationHead(combined_hidden_size, self.num_labels)
        
    def forward(self, input_ids,  seq_emb, text_emb, unimol_emb, kg_emb,  attention_mask, labels=None):
        
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        sequence_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        full_embeddings = torch.cat((sequence_output, text_emb, unimol_emb, kg_emb), dim=1)
        assert full_embeddings.shape[1] == 2112
                
        # following line gives IndexError: too many indices for tensor of dimension 2, how to fix?
        # logits = self.classifier(full_embeddings)
        # fixed line:
        logits = self.classifier(full_embeddings)
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            
        return outputs  # (loss), logits, (hidden_states), (attentions)


model_class = MultiModalTransformers_For_Classification
config_class = RobertaConfig
tokenizer_name = args.tokenizer_name

tokenizer_class = RobertaTokenizerFast
tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)

# Prepare and Get Data
class SELFIESTransfomers_Dataset(Dataset):
    def __init__(self, data, tokenizer, MAX_LEN):
        text, seq_emb, text_emb, unimol_emb, kg_emb = data
        
        self.examples = tokenizer(text=text, text_pair=None, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        self.seq_emb = seq_emb
        self.text_emb = torch.tensor(text_emb, dtype=torch.float)
        self.unimol_emb = torch.tensor(unimol_emb, dtype=torch.float)
        self.kg_emb = torch.tensor(kg_emb, dtype=torch.float)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        
        item = {key: self.examples[key][index] for key in self.examples}
        item['seq_emb'] = self.seq_emb[index]
        item['text_emb'] = self.text_emb[index]
        item['unimol_emb'] = self.unimol_emb[index]
        item['kg_emb'] = self.kg_emb[index]
        
        return item

pred_set = pd.read_pickle(args.pred_set)

MAX_LEN = 128

pred_examples = (pred_set['selfies'].astype(str).tolist(), 
                pred_set['sequence_embeddings'].tolist(),
                pred_set['text_embeddings'].tolist(),
                pred_set['unimol_embeddings'].tolist(),
                pred_set['kg_embeddings'].tolist()
                  )

pred_dataset = SELFIESTransfomers_Dataset(pred_examples, tokenizer, MAX_LEN)

training_args = torch.load(args.training_args)

model_name = args.model_name
config = config_class.from_pretrained(model_name, num_labels=2)
model = model_class.from_pretrained(model_name, config=config)

trainer = Trainer(model=model, args=training_args)  # the instantiated 🤗 Transformers model to be trained  # training arguments, defined above  # training dataset  # evaluation dataset
raw_pred, label_ids, metrics = trainer.predict(pred_dataset)
print('Raw pred:', raw_pred)
y_pred = np.argmax(raw_pred, axis=1).astype(int)

res = pd.concat([pred_set, pd.DataFrame(y_pred, columns=["prediction"])], axis = 1)

if not os.path.exists("data/predictions"):
    os.makedirs("data/predictions")

res = res.iloc[:, [0, -1]]
res.to_csv("data/predictions/{}_predictions.csv".format(args.dataset), index=False)
print("Predictions saved to data/predictions/{}_predictions.csv".format(args.dataset))