from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class BlizzardDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.blizzard_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.blizzard_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        blizzard = self.blizzard_frame.iloc[idx, :]
        blizzard = np.array([blizzard])
        blizzard = blizzard.astype('float').reshape(-1, 1)
        sample = {'blizzard': blizzard}
        return sample

def show_audio(blizzard):
    x = list(range(200))
    y = blizzard
    plt.plot(x,y)
    plt.show(block=True)

csv_filename = '/Users/mariamma/Documents/phd/rnn_project/code/dataset/X_train_blizzard.csv'
blizzard_dataset = BlizzardDataset(csv_file=csv_filename,
                                    root_dir=None)

# fig = plt.figure()
# for i in range(len(blizzard_dataset)):
#     sample = blizzard_dataset[i]

#     print(i, sample['blizzard'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_audio(**sample)

#     if i == 3:
#         plt.show()
#         break
train_dataloader = DataLoader(blizzard_dataset, batch_size=4,
                        shuffle=False, num_workers=4)


# Helper function to show a batch
def show_blizzard_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    blizzard_batch = sample_batched['blizzard']
    batch_size = len(blizzard_batch)
    print()
    
    for i in range(batch_size):
        x = list(range(200))
        y = blizzard_batch[i, :]
        print(blizzard_batch.size())
        print("Data : ", y.reshape(1,-1))
        plt.plot(x,y)
        plt.title('Batch from dataloader')
        plt.show(block=True)


# for i_batch, sample_batched in enumerate(train_dataloader):
#     print(i_batch, sample_batched['blizzard'].size())

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_blizzard_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show(block=True)
#         break


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def Gaussian(y, mu, sig):
    """
    Gaussian negative log-likelihood
    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    # nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
    #                   T.log(2 * np.pi), axis=-1)
    xmy = ((y - mu)**2).sum(2)
    K =  - 0.5* xmy / (s**2) -torch.log(sig)
    return K

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hn = torch.randn(2, 3, 20)
        self.cn = torch.randn(2, 3, 20)
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(800, 4000)
        self.linear_theta_stack = nn.Sequential(
            nn.Linear(4000, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU()
        )
        self.theta_sig = nn.Linear(800,200)
        self.theta_mu = nn.Softplus(800,200)
        

    def forward(self, x):
        x = self.flatten(x)
        linear_output = self.linear_relu_stack(x)
        rnn_output, (self.hn, self.cn) = self.rnn(linear_output, (self.hn, self.cn))
        theta_output = self.linear_theta_stack(hn)
        theta_mu_output = self.theta_mu(theta_output)
        theta_sig_output = self.theta_sig(theta_output)
        # self.recon = Gaussian(x, theta_mu_output, theta_sig_output)
        # self.recon_term = torch.mean(recon)
        return theta_mu_output, theta_sig_output, rnn_output

model = NeuralNetwork().to(device)
print(model)

# loss_fn = nn.CrossEntropyLoss()
loss_fn = None
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print("Size of dataloader dataset :: ", size)
    for batch, (X ) in enumerate(dataloader):
        X = X['blizzard'].to(device)

        # Compute prediction error
        print("X :: ", X.size())
        mu, sigma, rnn_output = model(X.float())
        print("Pred :: ", mu, sigma)
        loss = model.recon_term
        print("Loss :: ", loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model)
print("Done!")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")            