import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


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