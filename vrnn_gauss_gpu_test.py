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
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def KLGaussian(phi_mu_output, phi_sig_output, prior_mu_output, prior_sig_output):   
    batch_size = BATCH_SIZE
    mu1 = phi_mu_output.view(batch_size, 200)
    sig1 = phi_sig_output.view(batch_size, 200)
    mu2 = prior_mu_output.view(batch_size, 200)
    sig2 = prior_sig_output.view(batch_size, 200)

    sig1 = sig1.add(.000001)
    sig2 = sig2.add(.000001)
    sig2_log = torch.log(sig2)
    sig1_log = torch.log(sig1)
    res1 = sig2_log.sub(sig1_log)

    mu_sub = mu1.sub(mu2)
    mu_sub_pow = mu_sub.pow(2)
    sig1_pow = sig1.pow(2)
    sig2_pow = sig2.pow(2)
    mu_sig_add = sig1_pow.add(mu_sub_pow)
    sig2_pow = sig2_pow.add(.000000001)
    div_op = mu_sig_add.true_divide(sig2_pow.sub(1))
    res2 = div_op.mul(0.5)
    res3 = res1.add(res2)
    res = res3.mul(-1)
    print("Res1={}, res2={}, res3={}, res={}".format(res1.mean(), res2.mean(), res3.mean(), res.mean()))
    return res


def Gaussian(y, m, sigma, phi_mu_output, phi_sig_output, prior_mu_output, prior_sig_output):
    pi = 3.1415927410125732
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
    K1 =  xmy.add(sig_log)

    tensor = torch.ones((2,), dtype=torch.float64, device=y.device)
    pi_tensor = tensor.new_full((batch_size, 200), 2*pi)
    pi_log = torch.log(pi_tensor)
    K2 = K1.add(pi_log)
    K = K2.mul(0.5)
    print("K size :: ", K.size(), " K mean :: ", K.mean())
    res = KLGaussian(phi_mu_output, phi_sig_output, prior_mu_output, prior_sig_output)
    return K.add(res)
    
    

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.h0 = torch.randn(1, BATCH_SIZE, 4000, device="cuda")
        self.c0 = torch.randn(1, BATCH_SIZE, 4000, device="cuda")


        self.flatten = nn.Flatten()
        self.x_linear_stack = nn.Sequential(
            nn.Linear(200, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU()
        )
        self.phi_linear_stack = nn.Sequential(
            nn.Linear(4600, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        self.phi_mu = nn.Linear(500,200)
        self.phi_sig = nn.Sequential( 
                # nn.Softplus(800,800),
                nn.Linear(500,200),
                nn.ReLU()
                )
        self.prior_linear_stack = nn.Sequential(
            nn.Linear(4000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(500,200)
        self.prior_sig = nn.Sequential( 
                # nn.Softplus(800,800),
                nn.Linear(500,200),
                nn.ReLU()
                )
        self.z_linear_stack = nn.Sequential(
            nn.Linear(200, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(1200, 4000)
        self.theta_linear_stack = nn.Sequential(
            nn.Linear(4600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU()
        )
        self.theta_mu = nn.Linear(600,200)
        self.theta_sig = nn.Sequential( 
                # nn.Softplus(800,800),
                nn.Linear(600,200),
                nn.ReLU()
                )


    def forward(self, x, sn):

        x = self.flatten(x)
        # sn = torch.randn(1, BATCH_SIZE, 4000, device="cuda")
        x_linear_output = self.x_linear_stack(x)
        # print("x_linear_output :: ", x_linear_output.size())
        # print("sn :: ", sn.mean())

        phi_linear_input = torch.cat((x_linear_output, sn.view(BATCH_SIZE, 4000)),1)
        # print("phi_linear_input :: ", phi_linear_input.mean())
        phi_linear_output = self.phi_linear_stack(phi_linear_input)
        phi_mu_output = self.phi_mu(phi_linear_output)
        phi_sig_output = self.phi_sig(phi_linear_output)

        prior_linear_output = self.prior_linear_stack(sn.view(BATCH_SIZE, 4000))
        prior_mu_output = self.prior_mu(prior_linear_output)
        prior_sig_output = self.prior_sig(prior_linear_output)

        z_linear_stack = torch.normal(phi_mu_output, phi_sig_output)
        z_linear_output = self.z_linear_stack(z_linear_stack)

        # print("z_linear_output :: ", z_linear_output.mean())
        rnn_input = torch.cat((x_linear_output, z_linear_output),1)
        sn, (hn, cn) = self.rnn(rnn_input.view(-1, BATCH_SIZE, 1200), (self.h0, self.c0))
        # print("rnn_output :: ", sn.size())
        # self.sn = rnn_output.detach().clone()

        # print("sn :: ", sn.mean())
        # print("hn :: ", hn.mean())
        
        theta_input = torch.cat((z_linear_output, sn.view(BATCH_SIZE, 4000)),1)
        theta_output = self.theta_linear_stack(theta_input)
        theta_mu_output = self.theta_mu(theta_output)
        theta_sig_output = self.theta_sig(theta_output)
        return theta_mu_output, theta_sig_output, phi_mu_output, phi_sig_output, prior_mu_output, prior_sig_output, sn


model = NeuralNetwork().to(device)
print(model)

loss_fn = Gaussian
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
loss_value = list()


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    n = 4000
    rnn_arr = np.empty(shape=[0, n])
    loss_value = list()
    with torch.no_grad():
        # for X in dataloader:
        sn = torch.randn(1, BATCH_SIZE, 4000, device="cuda")
        for batch, (X ) in enumerate(dataloader):
            # print(batch, len(X))
            X = X['blizzard'].to(device)
            # print("X size :: ", X.size())
            theta_mu_output, theta_sig_output, phi_mu_output, phi_sig_output, prior_mu_output, prior_sig_output, sn = model(X.float(), sn)
            loss = Gaussian(X, theta_mu_output, theta_sig_output, phi_mu_output, phi_sig_output, prior_mu_output, prior_sig_output)
            loss_value.append(loss.mean())
            # print("rnn output :: ", rnn_output.size())
            rnn_out = sn.view(BATCH_SIZE,4000).cpu().numpy()
            rnn_arr = np.append(rnn_arr, rnn_out, axis=0)    
    np.savetxt("/home/mariamma/rnn_vae/code/vrnn_gauss_out.csv", rnn_arr, delimiter=',')       
    print("Mean :: ", sum(loss_value)/len(loss_value))

model_path = '/home/mariamma/rnn_vae/code/vrnn_model.pth'
model = torch.load(model_path)
test(test_dataloader, model)
