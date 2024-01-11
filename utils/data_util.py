import pandas as pd
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine

class HackerNewsBigrams(Dataset):
    """Hacker News dataset."""

    def __init__(self, train=True):
        """
        Arguments:
            train (bool): true if in training mode, false if in evaluation mode
        """
        self.train = train
        self.engine = create_engine(f'postgresql://{os.environ["DBUSER"]}:{os.environ["DBPW"]}@localhost:5432/hn')
        with self.engine.begin() as con:
            self.df = pd.read_sql(sql='SELECT text FROM comments LIMIT 100000', con=con)
            
        self.Xch = []
        self.ych = []
        for text in self.df['text'].str.lower():
            if text is not None:
                l = ['<>'] + list(text) + ['<>']
                for c1, c2 in zip(l, l[1:]):
                    if (((c1.isalpha()) & (c1.isascii())) | (c1 == '<>') | (c1 == ' ')) & (((c2.isalpha()) & (c2.isascii())) | (c2 == '<>') | (c2 == ' ')):
                        self.Xch.append(c1)
                        self.ych.append(c2)

        self.chars = set(self.Xch)
        self.ctoi = {c:i for i, c in enumerate(sorted(self.chars))}
        self.itoc = {i:c for c, i in self.ctoi.items()}

        self.X = torch.tensor([self.ctoi[x] for x in self.Xch])
        self.X = F.one_hot(self.X, num_classes=28).float()
        self.y = torch.tensor([self.ctoi[y] for y in self.ych])
        
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(self.X, self.y, test_size=0.1, random_state=42)


    def __len__(self):
        if self.train:
            return len(self.Xtr)
        else:
            return len(self.Xte)

    def __getitem__(self, idx):
        if self.train:
            return (self.Xtr[idx], self.ytr[idx])
        else:
            return (self.Xte[idx], self.yte[idx])
        

class HackerNewsContext(Dataset):
    """Hacker News dataset."""

    def __init__(self, train=True, context_size=3, number_of_strings=1000):
        """
        Arguments:
            train (bool): true if in training mode, false if in evaluation mode
            context_size (int): number of characters in context
        """
        self.train = train
        self.context_size = context_size
        self.number_of_strings = number_of_strings
        self.engine = create_engine(f'postgresql://{os.environ["DBUSER"]}:{os.environ["DBPW"]}@localhost:5432/hn')
        with self.engine.begin() as con:
            self.df = pd.read_sql(sql=f'SELECT text FROM comments LIMIT {self.number_of_strings}', con=con)
            
        self.contexts = []
        self.ys = []
        for text in self.df['text'].str.lower():
            if text is not None:
                context = ['<>'] * context_size
                l = list(text) + ['<>']
                for c in l:
                    if (((c.isalpha()) & (c.isascii())) | (c == '<>') | (c == ' ')):
                        self.contexts.append(context)
                        self.ys.append(c)
                        context = context[1:] + [c]

        self.chars = set(self.ys)
        self.ctoi = {c:i for i, c in enumerate(sorted(self.chars))}
        self.itoc = {i:c for c, i in self.ctoi.items()}

        self.X_indexes = torch.tensor([[self.ctoi[char] for char in context] for context in self.contexts])
        self.X = F.one_hot(self.X_indexes, num_classes=28).float()
        self.y = torch.tensor([self.ctoi[y] for y in self.ys])

        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(self.X, self.y, test_size=0.1, random_state=42)


    def __len__(self):
        if self.train:
            return len(self.Xtr)
        else:
            return len(self.Xte)

    def __getitem__(self, idx):
        if self.train:
            return (self.Xtr[idx], self.ytr[idx])
        else:
            return (self.Xte[idx], self.yte[idx])