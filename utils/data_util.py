import pandas as pd
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine

class HousingPriceDataset(Dataset):
    """Housing price dataset."""

    def __init__(self, csv_file, train=True):
        """
        Arguments:
            csv_file (string): Path to the csv file with housing price data.
        """
        self.train = train
        self.df = pd.read_csv('housing_price_dataset.csv')
        # print(f'Missing values:\n{self.df.isna().sum()}')
        self.ydf = self.df[['Price']]
        self.Xdf = self.df.drop(columns=['Price'])
        # print(f'\nShapes:\nX: {self.Xdf.shape}\ny: {self.ydf.shape}')
        self.Xdf_tr, self.Xdf_te, self.ydf_tr, self.ydf_te = train_test_split(self.Xdf, self.ydf, test_size=0.1, random_state=42)
        # print(f'Shapes:\nXtr: {self.Xdf_tr.shape}\nXte: {self.Xdf_te.shape}\nytr: {self.ydf_tr.shape}\nyte: {self.ydf_te.shape}')
        self.preprocessor = ColumnTransformer(
            [("Categorical", OneHotEncoder(), ['Bedrooms', 'Bathrooms', 'Neighborhood']),
            ("Numerical", StandardScaler(), ['SquareFeet', 'YearBuilt'])]
        )
        self.target_scaler = StandardScaler()

        self.Xtr = torch.Tensor(self.preprocessor.fit_transform(self.Xdf_tr))
        self.Xte = torch.Tensor(self.preprocessor.transform(self.Xdf_te))
        self.ytr = torch.Tensor(self.target_scaler.fit_transform(self.ydf_tr))
        self.yte = torch.Tensor(self.target_scaler.transform(self.ydf_te))

        # print(f'Shapes:\nXtr: {self.Xtr.shape}\nXte: {self.Xte.shape}\nytr: {self.ytr.shape}\nyte: {self.yte.shape}')

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
            self.df = pd.read_sql(sql='SELECT text FROM comments ORDER BY random() LIMIT 100000', con=con)
            
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

    def __init__(self, train=True, context_size=3):
        """
        Arguments:
            train (bool): true if in training mode, false if in evaluation mode
            context_size (int): number of characters in context
        """
        self.train = train
        self.context_size = context_size
        self.engine = create_engine(f'postgresql://{os.environ["DBUSER"]}:{os.environ["DBPW"]}@localhost:5432/hn')
        with self.engine.begin() as con:
            self.df = pd.read_sql(sql='SELECT text FROM comments ORDER BY random() LIMIT 50000', con=con)
            
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