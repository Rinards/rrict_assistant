import os
from transformers import AutoTokenizer, AutoModel
import torch

import torch.nn as nn
import torch.nn.functional as F

# Use AdamW optimizer
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
# Model labels should already be loaded by this point

# Load model from HuggingFace Hub
MODEL_NAME = 'sentence-transformers/paraphrase-albert-small-v2'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
albert_model = AutoModel.from_pretrained(MODEL_NAME)

@dataclass
class Config:
    batch_size: int = 16
    shuffle: bool = True
    epochs: int = 3
    seed: int = 20
    lr: float = 0.0017

def get_optimizer(model, lr):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.albert.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": lr / 10,  # Use a lower learning rate for ALBERT
        },
        {
            "params": [p for n, p in model.albert.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr / 10,  # Use a lower learning rate for ALBERT
        },
        {
            "params": [p for n, p in model.named_parameters() if "albert" not in n],
            "weight_decay": 0.01,
            "lr": lr,  # Use the original learning rate for the classifier layers
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Net(nn.Module):
    @staticmethod
    def pretrained(class_num: int, model_path: str = "./MODEL", device=torch.device("cpu")):
        model = Net(class_num)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def __init__(self, class_num: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.albert = albert_model

        self.l2 = mean_pooling
        self.dropout = nn.Dropout(p=0.23)

        self.l3 = nn.Linear(self.albert.config.hidden_size, 68)
        self.l4 = nn.Linear(68, class_num)

        self.l3.requires_grad = True
        self.l4.requires_grad = True

        for name, param in self.albert.named_parameters():
            param.requires_grad = False
        #     param.requires_grad = True

    def forward(self, x):
        x = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt').to(next(self.parameters()).device)

        attention_mask = x['attention_mask']

        x = self.albert(**x)
        x = self.l2(x, attention_mask)
        x = F.sigmoid(self.l3(x))
        x = F.softmax(self.l4(x), dim=1)
        return x

    def from_tokenized(self, x):
        attention_mask = x['attention_mask']

        x = self.albert(**x)
        x = self.l2(x, attention_mask)
        x = self.dropout(x)
        x = F.linear(self.l3(x))
        x = F.softmax(self.l4(x), dim=1)
        return x

def train(model, data_loader, conf=Config(), device='cpu'):
    if type(device) is str:
        device = torch.device(device)

    model.to(device)
    pid = os.getpid()

    torch.manual_seed(conf.seed)
    # optimizer = AdamW(model.parameters(), lr=conf.lr)
    optimizer = get_optimizer(model, lr=conf.lr)

    # loss_fn = F.nll_loss
    # loss = nn.L1loss(output, Y)
    loss_fn = nn.CrossEntropyLoss().to(device)

    model.train()
    for epoch in range(conf.epochs):
        count = 0
        for data in data_loader:
            optimizer.zero_grad()

            X = data['review_text']
            Y = data['targets'].to(device)

            output = model(X)

            loss = loss_fn(output, Y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if count % 10 == 0:
                print(f"({pid}) "
                      f"Epoch: {epoch}, "
                      f"tdata[{count}] | "
                      f"Loss {round(loss.item(), 6)}")
            count += 1

def run_model(model, sentences, model_labels):
    labels = []
    Y = model(sentences).detach().numpy()
    for label_tk in Y:
        labels.append(model_labels[label_tk.argmax()])
    return labels

if __name__ == '__main__':
    # Sentences we want sentence embeddings for
    sentences = ['What time is it?', 'Give me the time']

    model = Net()
    Y = model(sentences).detach().numpy()
    print(Y)
