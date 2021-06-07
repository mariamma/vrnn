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
        return 62336 #len(self.blizzard_frame)

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

csv_filename = '/home/mariamma/rnn_vae/dataset/X_test_blizzard.csv'
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
test_dataloader = DataLoader(blizzard_dataset, batch_size=BATCH_SIZE,
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

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    n = 4000
    rnn_arr = np.empty(shape=[0, n])
    with torch.no_grad():
        # for X in dataloader:
        for batch, (X ) in enumerate(dataloader):
            # print(batch, len(X))
            X = X['blizzard'].to(device)
            # print("X size :: ", X.size())
            theta_mu_output, theta_sig_output, rnn_output = model(X.float())
            # print("rnn output :: ", rnn_output.size())
            rnn_out = rnn_output.view(BATCH_SIZE,4000).cpu().numpy()
            rnn_arr = np.append(rnn_arr, rnn_out, axis=0)    
    np.savetxt("/home/mariamma/rnn_vae/code/rnn_gauss_out.csv", rnn_arr, delimiter=',')       
    
model_path = '/home/mariamma/rnn_vae/code/rnn_model.pth'
model = torch.load(model_path)
test(test_dataloader, model)
