import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss /= batch
    return train_loss

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss


class EmbeddingsNN:
    def __init__(self, embedding_dims, context_size, batch_size, device):
        self.batch_size = batch_size
        self.embedding_dims = embedding_dims
        self.context_size = context_size
        self.device = device
        self.g = torch.Generator(device=self.device).manual_seed(0)
        self.C = torch.randn((28, self.embedding_dims), generator=self.g, device=self.device) * 0.01# embedding lookup-table
        self.W1 = torch.randn((self.context_size*self.embedding_dims, 256), generator=self.g, device=self.device) * (2/(self.context_size*self.embedding_dims))**0.5 # weights hidden layer
        self.b1 = torch.randn(256, generator=self.g, device=self.device) * 0.01 # biases hidden layer
        self.W2 = torch.randn((256, 28), generator=self.g, device=self.device) * (2/256)**0.5 # weights output layer
        self.b2 = torch.randn(28, generator=self.g, device=self.device) * 0.01 # biases output layer
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, X, y):
        self.embedding = (X @ self.C).view(self.batch_size, self.context_size*self.embedding_dims)
        self.h = torch.relu(self.embedding @ self.W1 + self.b1) # (batch_size, 256)
        self.logits = self.h @ self.W2 + self.b2 # (batch_size, 28)
        self.loss = F.cross_entropy(self.logits, y)
        return self.loss.item()
        
    def backward(self):
        for p in self.parameters:
            p.grad = None
        self.loss.backward()

    def update_params(self, lr):
        for p in self.parameters:
            p.data += -lr * p.grad

    def sample(self, ctoi, itoc):
        text = ''
        n = 0
        nll = 0.0
        context = ['<>'] * self.context_size

        while True:
            X = F.one_hot(torch.tensor([ctoi[c] for c in context], device=self.device), num_classes=28).float()
            
            # forward pass
            with torch.no_grad():
                self.embedding = (X @ self.C).view(self.context_size*self.embedding_dims)
                self.h = torch.relu(self.embedding @ self.W1 + self.b1) 
                self.logits = self.h @ self.W2 + self.b2 
                self.probs = F.softmax(self.logits, dim=0)
                ix = torch.multinomial(self.probs, num_samples=1, replacement=True, generator=self.g).item()


            # break if end-character was sampled
            if ix==ctoi['<>']:
                break
            
            # update params
            context = context[1:] + [itoc[ix]]
            text += itoc[ix]
            n += 1

            # calculate negative log loss
            p = self.probs[ix].log().item()
            nll -= p
        return text, nll/n