from __future__ import print_function, division
import os
import torch
import pandas as pd
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

BATCH_SIZE = 64

class BlizzardDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.blizzard_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 390400 #len(self.blizzard_frame)

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

csv_filename = '/home/mariamma/rnn_vae/dataset/X_train_blizzard.csv'
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
train_dataloader = DataLoader(blizzard_dataset, batch_size=BATCH_SIZE,
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
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def Gaussian(y, m, sigma):
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
    batch_size = BATCH_SIZE
    x = y.view(batch_size, 200)
    mu = m.view(batch_size, 200)
    sig = sigma.view(batch_size, 200)
    # print("X mean :: ", x.mean())
    # print("Mu mean :: ", mu.mean())
    # print("Sigma mean :: ", sig.mean())
    x_mu_diff = x.sub(mu)
    x_diff_pow = x_mu_diff.pow(2)
    # print("X mu diff pow :: ", x_diff_pow.mean())
    sig_pow = sig.pow(2).add(.000001)
    # print("sig_pow :: ", sig_pow.mean())
    xmy =  x_diff_pow.true_divide(sig_pow)
    sig = sig.add(.000001)
    sig_log = torch.log(sig).mul(2)
    # print("Xmy ::  ", xmy.size(), " Xmy mean :: ", xmy.mean())
    # print("sig size :: ", sig_log.size(), " sig mean :: ", sig_log.mean())
    K =  xmy.add(sig_log)
    # print("K size :: ", K.size(), " K mean :: ", K.mean())
    return K

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
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
        self.theta_mu = nn.Linear(800,200)
        self.theta_sig = nn.Sequential( 
                # nn.Softplus(800,800),
                nn.Linear(800,200),
                nn.ReLU()
                )
        


    def forward(self, x):
        x = self.flatten(x)
        h0 = torch.randn(1, BATCH_SIZE, 4000, device=x.device)
        c0 = torch.randn(1, BATCH_SIZE, 4000, device=x.device)
        linear_output = self.linear_relu_stack(x)
        rnn_output, (hn, cn) = self.rnn(linear_output.view(-1, BATCH_SIZE, 800), (h0, c0))
        # print("hn :: ", hn.mean())
        # print("cn :: ", cn.mean())
        # print("rnn_output :: ", rnn_output.mean())
        theta_output = self.linear_theta_stack(rnn_output)
        theta_mu_output = self.theta_mu(theta_output)
        theta_sig_output = self.theta_sig(theta_output)
        # self.recon = Gaussian(x, theta_mu_output, theta_sig_output)
        # self.recon_term = torch.mean(recon)
        return theta_mu_output, theta_sig_output, rnn_output

model = NeuralNetwork().to(device)
print(model)

# loss_fn = nn.CrossEntropyLoss()
loss_fn = Gaussian
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
loss_value = list()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print("Size of dataloader dataset :: ", size)
    for batch, (X ) in enumerate(dataloader):
        X = X['blizzard'].to(device)
        # print("X :: ", X.mean())

        # Compute prediction error
        mu, sigma, rnn_output = model(X.float())
        # print("Size of mu and sigma :: ", mu.mean(), sigma.mean())
        loss = loss_fn(X, mu, sigma)
        # print("Loss :: ", loss.mean())

        # Backpropagation
        optimizer.zero_grad()
        loss.mean().backward(retain_graph=True)
        optimizer.step()

        if batch % 100 == 0:
            print(batch, len(X))
            loss, current = loss.mean(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 
        if batch % 1000 == 0:              
            loss_value.append(loss)  

epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model)
print("Done!")

model_path = '/home/mariamma/rnn_vae/code/rnn_model.pth'
torch.save(model, model_path)
loss_np = np.array(loss_value)
np.savetxt("/home/mariamma/rnn_vae/code/loss.csv", loss_np, delimiter=',')



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