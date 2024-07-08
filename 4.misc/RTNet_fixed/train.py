import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Check Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

AlexTransform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=True, download=True, transform=AlexTransform),
        batch_size=500, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=False, download=True, transform=AlexTransform),
        batch_size=100, shuffle=False)

# AlexNet
class alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        # out = F.log_softmax(out, dim=1)

        return out

# Define Hyper parameters
img_size = 28 * 28
hidden_layer_size = 1024
num_classes = 10

net = alexnet().to(device)
# softmax
log_softmax = nn.LogSoftmax(dim=1)


def model(x_data, y_data):

    convLayer1_w = Normal(loc=torch.ones_like(net.conv1[0].weight), scale=torch.ones_like(net.conv1[0].weight))
    convLayer1_b = Normal(loc=torch.ones_like(net.conv1[0].bias), scale=torch.ones_like(net.conv1[0].bias))

    convLayer2_w = Normal(loc=torch.ones_like(net.conv2[0].weight), scale=torch.ones_like(net.conv2[0].weight))
    convLayer2_b = Normal(loc=torch.ones_like(net.conv2[0].bias), scale=torch.ones_like(net.conv2[0].bias))

    convLayer3_w = Normal(loc=torch.ones_like(net.conv3[0].weight), scale=torch.ones_like(net.conv3[0].weight))
    convLayer3_b = Normal(loc=torch.ones_like(net.conv3[0].bias), scale=torch.ones_like(net.conv3[0].bias))

    convLayer4_w = Normal(loc=torch.ones_like(net.conv4[0].weight), scale=torch.ones_like(net.conv4[0].weight))
    convLayer4_b = Normal(loc=torch.ones_like(net.conv4[0].bias), scale=torch.ones_like(net.conv4[0].bias))

    convLayer5_w = Normal(loc=torch.ones_like(net.conv5[0].weight), scale=torch.ones_like(net.conv5[0].weight))
    convLayer5_b = Normal(loc=torch.ones_like(net.conv5[0].bias), scale=torch.ones_like(net.conv5[0].bias))

    fc1Layer_w = Normal(loc=torch.ones_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1Layer_b = Normal(loc=torch.ones_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

    fc2Layer_w = Normal(loc=torch.ones_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
    fc2Layer_b = Normal(loc=torch.ones_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))

    fc3Layer_w = Normal(loc=torch.ones_like(net.fc3.weight), scale=torch.ones_like(net.fc3.weight))
    fc3Layer_b = Normal(loc=torch.ones_like(net.fc3.bias), scale=torch.ones_like(net.fc3.bias))

    priors = {'conv1[0].weight': convLayer1_w,
              'conv1[0].bias': convLayer1_b,
              'conv2[0].weight': convLayer2_w,
              'conv2[0].bias': convLayer2_b,
              'conv3[0].weight': convLayer3_w,
              'conv3[0].bias': convLayer3_b,
              'conv4[0].weight': convLayer4_w,
              'conv4[0].bias': convLayer4_b,
              'conv5[0].weight': convLayer5_w,
              'conv5[0].bias': convLayer5_b,
              'fc1.weight': fc1Layer_w,
              'fc1.bias': fc1Layer_b,
              'fc2.weight': fc2Layer_w,
              'fc2.bias': fc2Layer_b,
              'fc3.weight': fc3Layer_w,
              'fc3.bias': fc3Layer_b}

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))

    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
    # pyro.sample("obs",,obs = y_data)



softplus = torch.nn.Softplus()

def guide(x_data, y_data):

    # First layer weight distribution priors
    convLayer1w_mu    = torch.randn_like(net.conv1[0].weight)
    convLayer1w_sigma = torch.randn_like(net.conv1[0].weight)
    convLayer1w_mu_param    = pyro.param("convLayer1w_mu", convLayer1w_mu)
    convLayer1w_sigma_param = softplus(pyro.param("convLayer1w_sigma", convLayer1w_sigma))
    convLayer1_w = Normal(loc=convLayer1w_mu_param, scale=convLayer1w_sigma_param)

    # First layer bias distribution priors
    convLayer1b_mu    = torch.randn_like(net.conv1[0].bias)
    convLayer1b_sigma = torch.randn_like(net.conv1[0].bias)
    convLayer1b_mu_param    = pyro.param("convLayer1b_mu", convLayer1b_mu)
    convLayer1b_sigma_param = softplus(pyro.param("convLayer1b_sigma", convLayer1b_sigma))
    convLayer1_b = Normal(loc=convLayer1b_mu_param, scale=convLayer1b_sigma_param)

    # Second layer weight distribution priors
    convLayer2w_mu    = torch.randn_like(net.conv2[0].weight)
    convLayer2w_sigma = torch.randn_like(net.conv2[0].weight)
    convLayer2w_mu_param    = pyro.param("convLayer2w_mu", convLayer2w_mu)
    convLayer2w_sigma_param = softplus(pyro.param("convLayer2w_sigma", convLayer2w_sigma))
    convLayer2_w = Normal(loc=convLayer2w_mu_param, scale=convLayer2w_sigma_param)

    # Second layer bias distribution priors
    convLayer2b_mu    = torch.randn_like(net.conv2[0].bias)
    convLayer2b_sigma = torch.randn_like(net.conv2[0].bias)
    convLayer2b_mu_param    = pyro.param("convLayer2b_mu", convLayer2b_mu)
    convLayer2b_sigma_param = softplus(pyro.param("convLayer2b_sigma", convLayer2b_sigma))
    convLayer2_b = Normal(loc=convLayer2b_mu_param, scale=convLayer2b_sigma_param)

    # Third layer weight distribution priors
    convLayer3w_mu    = torch.randn_like(net.conv3[0].weight)
    convLayer3w_sigma = torch.randn_like(net.conv3[0].weight)
    convLayer3w_mu_param    = pyro.param("convLayer3w_mu", convLayer3w_mu)
    convLayer3w_sigma_param = softplus(pyro.param("convLayer3w_sigma", convLayer3w_sigma))
    convLayer3_w = Normal(loc=convLayer3w_mu_param, scale=convLayer3w_sigma_param)

    # Third layer bias distribution priors
    convLayer3b_mu    = torch.randn_like(net.conv3[0].bias)
    convLayer3b_sigma = torch.randn_like(net.conv3[0].bias)
    convLayer3b_mu_param    = pyro.param("convLayer3b_mu", convLayer3b_mu)
    convLayer3b_sigma_param = softplus(pyro.param("convLayer3b_sigma", convLayer3b_sigma))
    convLayer3_b = Normal(loc=convLayer3b_mu_param, scale=convLayer3b_sigma_param)

    # Fourth layer weight distribution priors
    convLayer4w_mu    = torch.randn_like(net.conv4[0].weight)
    convLayer4w_sigma = torch.randn_like(net.conv4[0].weight)
    convLayer4w_mu_param    = pyro.param("convLayer4w_mu", convLayer4w_mu)
    convLayer4w_sigma_param = softplus(pyro.param("convLayer4w_sigma", convLayer4w_sigma))
    convLayer4_w = Normal(loc=convLayer4w_mu_param, scale=convLayer4w_sigma_param)

    # Fourth layer bias distribution priors
    convLayer4b_mu    = torch.randn_like(net.conv4[0].bias)
    convLayer4b_sigma = torch.randn_like(net.conv4[0].bias)
    convLayer4b_mu_param    = pyro.param("convLayer4b_mu", convLayer4b_mu)
    convLayer4b_sigma_param = softplus(pyro.param("convLayer4b_sigma", convLayer4b_sigma))
    convLayer4_b = Normal(loc=convLayer4b_mu_param, scale=convLayer4b_sigma_param)

    # Fifth layer weight distribution priors
    convLayer5w_mu    = torch.randn_like(net.conv5[0].weight)
    convLayer5w_sigma = torch.randn_like(net.conv5[0].weight)
    convLayer5w_mu_param    = pyro.param("convLayer5w_mu", convLayer5w_mu)
    convLayer5w_sigma_param = softplus(pyro.param("convLayer5w_sigma", convLayer5w_sigma))
    convLayer5_w = Normal(loc=convLayer5w_mu_param, scale=convLayer5w_sigma_param)

    # Fifth layer bias distribution priors
    convLayer5b_mu    = torch.randn_like(net.conv5[0].bias)
    convLayer5b_sigma = torch.randn_like(net.conv5[0].bias)
    convLayer5b_mu_param    = pyro.param("convLayer5b_mu", convLayer5b_mu)
    convLayer5b_sigma_param = softplus(pyro.param("convLayer5b_sigma", convLayer5b_sigma))
    convLayer5_b = Normal(loc=convLayer5b_mu_param, scale=convLayer5b_sigma_param)

    # First fully connected layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1Layer_w = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param).independent(1)

    # First fully connected layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1Layer_b = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    # Second fully connected layer weight distribution priors
    fc2w_mu = torch.randn_like(net.fc2.weight)
    fc2w_sigma = torch.randn_like(net.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2Layer_w = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param).independent(1)

    # Second fully connected layer bias distribution priors
    fc2b_mu = torch.randn_like(net.fc2.bias)
    fc2b_sigma = torch.randn_like(net.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2Layer_b = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)

    # Third fully connected layer weight distribution priors
    fc3w_mu = torch.randn_like(net.fc3.weight)
    fc3w_sigma = torch.randn_like(net.fc3.weight)
    fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
    fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", fc3w_sigma))
    fc3Layer_w = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param).independent(1)

    # Third fully connected layer bias distribution priors
    fc3b_mu = torch.randn_like(net.fc3.bias)
    fc3b_sigma = torch.randn_like(net.fc3.bias)
    fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
    fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", fc3b_sigma))
    fc3Layer_b = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)

    priors = {'conv1[0].weight': convLayer1_w,
              'conv1[0].bias': convLayer1_b,
              'conv2[0].weight': convLayer2_w,
              'conv2[0].bias': convLayer2_b,
              'conv3[0].weight': convLayer3_w,
              'conv3[0].bias': convLayer3_b,
              'conv4[0].weight': convLayer4_w,
              'conv4[0].bias': convLayer4_b,
              'conv5[0].weight': convLayer5_w,
              'conv5[0].bias': convLayer5_b,
              'fc1.weight': fc1Layer_w,
              'fc1.bias': fc1Layer_b,
              'fc2.weight': fc2Layer_w,
              'fc2.bias': fc2Layer_b,
              'fc3.weight': fc3Layer_w,
              'fc3.bias': fc3Layer_b}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()

def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return torch.argmax(mean, dim=1)


optim = Adam({'lr': 0.001})
svi = SVI(model, guide, optim, loss=Trace_ELBO())


num_iterations = 1
loss = 0
num_samples = 3
path = ''
acc_temp = 0

for j in range(num_iterations):
    net.train()
    loss = 0
    for batch_id, data in enumerate(tqdm(train_loader)):
        # calculate the loss and take a gradient step
        # print(data[0].shape())
        loss += svi.step(data[0].to(device), data[1].to(device))
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train

    print("Epoch ", j, " Loss ", total_epoch_loss_train)

    # if j < 15:
    #     continue

    net.eval()
    print('Prediction when network is forced to predict')
    correct = 0
    total = 0

    all_labels = []
    all_predicted = []
    for j, data in enumerate(tqdm(test_loader)):
        images, labels = data
        predicted = predict(images.to(device))
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()
        all_labels.append(labels)
        all_predicted.append(predicted)
    print("accuracy: %d %%" % (100 * correct / total))

    model_num = '01'
    if correct >= acc_temp:
        acc_temp = correct
        # torch.save({"model" : net.state_dict(), "guide" : guide}, path + "Epoch" + str(j) + "model_" + model_num + ".pt")
        # pyro.get_param_store().save(path + "Epoch" + str(j) + "model_" + model_num + "_params.pt")
        
        
        
        
def give_uncertainities(x, num_samples=1):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    # yhats = [F.log_softmax(model(x).data, dim=1) for model in sampled_models]
    #TODO:x为1,1,227,227 model为alexnet
    for model in sampled_models:
        b = model(x)#TODO：1,10 Tensor
        a = b.data
        yhats = [F.log_softmax(a, dim=1)]
        print("yhat")
        print(yhats)
    return yhats[0]

def compute_evidence(image):
    # image = image.unsqueeze(dim=0)
    y = give_uncertainities(image)
    return y

def compute_confidence(evidence):
    conf = torch.exp(evidence)
    conf = conf / torch.sum(conf)
    conf_diff = conf.sort().values[0][-1] - conf.sort().values[0][-2]
    return conf_diff, conf


def decide(image, threshold=10):
    rt = 0
    max_evidence, total_evidence = 0, 0
    while max_evidence < threshold:
      evidence = torch.exp(compute_evidence(image))
      total_evidence = total_evidence + evidence
      max_evidence = torch.max(total_evidence)
      print("max_evi")
      print(max_evidence)
      rt = rt + 1
    print("rt and evi")
    print(rt,total_evidence)
    choice = torch.argmax(total_evidence)
    confidence, confidence_array = compute_confidence(total_evidence)
    return int(choice.cpu().numpy()), rt, float(confidence.cpu().numpy()), confidence_array.cpu().numpy()


def save_df(Choice, RT, Confidence, Confidence_array, Threshold, Noise, Labels, path):
    simulations = {#'mnist_index': index,
                   'choice': Choice,
                   'rt': RT,
                   'confidence': Confidence,
                   'confidence array': Confidence_array,
                   'threshold': Threshold,
                   'noise': Noise,
                   'true label': Labels}
    df = pd.DataFrame(simulations)
    df.to_csv(path)
    return df


threshold_levels = [3, 5]
noise_levels = [2.1, 2.9]
save_path = 'RT_save' # path to where you want to save the output TODO:need to fix
all_choice, all_rt, all_confidence, all_confidence_array, all_threshold, all_labels, all_noise = [], [], [], [], [], [], []
for a in threshold_levels:
    print("Set the threshold to {}".format(str(a)))
    for noise in noise_levels:
        print('   Set the noise level to {}'.format(str(noise)))
        for i, (image, label) in enumerate(tqdm(test_loader)):
            choice, rt, confidence, confidence_array = decide((image + noise * torch.rand(image.shape)).to(device), threshold=a)
            all_choice.append(choice)
            all_rt.append(rt)
            all_confidence.append(confidence)
            all_confidence_array.append(confidence_array)
            all_threshold.append(a)
            all_noise.append(noise)
            all_labels.append(int(label.cpu().numpy()))
            # if (i%1000)==0:
            #     print('           {}'.format(str(i)))
        save_df(all_choice, all_rt, all_confidence, all_confidence_array, all_threshold, all_noise, all_labels, save_path+'/'+str(a)+'_'+str(noise)+'.csv')




