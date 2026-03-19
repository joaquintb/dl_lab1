#!/usr/bin/env python
# coding: utf-8

# ## Previous
# - The kernel of the script must be associated to an environment with pytorch installed
# - For example: master-dl
#     -   % Create a special kernel for the conda environment
#     -   conda install ipykernel
#     -   % configure Jupyter to use the Python kernel
#     -   ipython kernel install --user --name=master-dl
# - Other things
#     - Capturing the output of a notebook cell:
#         https://notebook.community/lifeinoppo/littlefishlet-scode/RES/REF/python_sourcecode/ipython-master/examples/IPython%20Kernel/Capturing%20Output
# 

# In[1]:


import torch
import torchvision

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#import torch.nn.functional as F
from torch.nn.functional import relu
from torch import nn
from torch import optim

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from tqdm import tqdm

import sys, time

# In[2]:


print('torch version:',torch.__version__)
print('torchvision version:',torchvision.__version__)


# ## Data

# #### Load MNIST data

# In[3]:


train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
test_dataset  = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())


# In[4]:


train_dataset, test_dataset


# In[5]:


# Some information of the dataset
print('Number of train examples:', train_dataset.data.shape[0])
print('Size of train examples:', train_dataset.data.shape[1:])
print('Range of input values:',torch.min(train_dataset.data),torch.max(train_dataset.data))
print('Classes:',train_dataset.classes)


# In[6]:


#InputSize  = 784                          # 28x28 pixels (not directly used in CNN)
   # np.prod(train_dataset.data.shape[1:])
NumClasses = len(train_dataset.classes)   # digits 0-9


# ## Architecture

# In[7]:


# Two convolutional layers + an MLP with one hidden layer (two fully connected layers)

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 8 output features (num_classes)
        self.fc1 = nn.Linear(16 * 7 * 7, 8)

        # Fully connected layer: 8 input features, 10 output features (num_classes)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = relu(self.conv1(x))      # Apply first convolution and ReLU activation
        x = self.pool1(x)              # Apply max pooling
        x = relu(self.conv2(x))      # Apply second convolution and ReLU activation
        x = self.pool2(x)              # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)                # Apply fully connected layer
        x = self.fc2(x)                # Apply fully connected layer
        return x


# In[8]:


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=1, num_classes=NumClasses).to(device)


# ## Training setting

# In[9]:


#help(torch.optim.SGD)


# In[10]:


#help(torch.optim.Adam)


# In[11]:


#
# Training parameters
#
LearningRate = 0.01
Momentum     = 0.0

BatchSize = 128
NEpochs   = 200

#
# Optimizer, loss and metrics
#
Optimizer = optim.SGD(model.parameters(), lr=LearningRate, momentum=Momentum)
#optimizer = optim.Adam(model.parameters(), lr=LearningRate)

Loss = nn.CrossEntropyLoss()


# #### Create DataLoaders

# In[12]:


train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=BatchSize, shuffle=True)


# In[14]:


# Some information of a batch
for batch_index, (inputs, targets) in enumerate(train_loader):
    # Check sizes of dataset
    print('Number of train examples:', inputs.shape[0])
    print('  *** NOTE THAT INPUT VALUES ARE ALREADY NORMALIZED ***')
    print('Range of input values:',torch.min(inputs),torch.max(inputs))
    print('Type of input values:',type(inputs),type(inputs[0,0,0]))
    print('Targets:',targets)
    break


# ## Training

# In[15]:


def model_train(device, model, data_loader, loss_fn, optimizer, epoch):
    model.train()
    
    train_loss = 0.0
    correct = 0
    for batch_index, (inputs, targets) in enumerate(data_loader):  ### tqdm(data_loader)
        # Move data and targets to the device (GPU/CPU)
        inputs  = inputs.to(device)
        targets = targets.to(device)

        # Forward pass: compute the model output
        scores = model(inputs)
        loss_b = loss_fn(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss_b.backward()

        # Optimization step: update the model parameters
        optimizer.step()

        # Statistics
        train_loss += loss_b.item()
        train_pred  = scores.argmax(dim=1, keepdim=True)
        correct    += train_pred.eq(targets.view_as(train_pred)).sum().item()
        if batch_index % 100 == 99:    # print every 100 mini-batches
            print(f'    [{epoch + 1}, {batch_index + 1:5d}] batch loss: {train_loss/100:.3f}')
            train_loss = 0.0

    train_loss    /= len(data_loader.dataset)
    train_accuracy = 100. * correct / len(data_loader.dataset)
    
    return train_loss, train_accuracy


def model_eval(device, model, data_loader, loss_fn):
    model.eval()
    eval_loss   = 0.0
    correct     = 0
    eval_preds  = []
    eval_labels = []
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(data_loader):
            # Move data and targets to the device (GPU/CPU)
            inputs  = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(inputs)
            loss_b = loss_fn(scores, targets)

            # Statistics
            eval_loss += loss_b.item()
            eval_pred  = scores.argmax(dim=1, keepdim=True)
            eval_preds.append(eval_pred.tolist())
            eval_labels.append(targets.view_as(eval_pred))
            correct   += eval_pred.eq(targets.view_as(eval_pred)).sum().item()

    eval_loss    /= len(data_loader.dataset)
    eval_accuracy = 100. * correct / len(data_loader.dataset)
    
    return eval_loss, eval_accuracy, eval_labels, eval_preds


# In[ ]:





# In[16]:


def model_fit(device, model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10):
    history = dict()
    history['train_loss'] = []
    history['train_accuracy'] = []
    history['val_loss'] = []
    history['val_accuracy'] = []

    train_loss, train_accuracy, _, _ = model_eval(device, model, train_loader, loss_fn)
    val_loss, val_accuracy, _, _     = model_eval(device, model, val_loader, loss_fn)
    history['train_loss'].append(train_loss);  history['train_accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss);      history['val_accuracy'].append(val_accuracy)
    
    for epoch in range(num_epochs):        
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss, train_accuracy   = model_train(device, model, train_loader, loss_fn, optimizer, epoch)
        val_loss, val_accuracy, _, _ = model_eval(device, model, val_loader, loss_fn)
            
        # print statistics
        print(f'  training accuracy: {train_accuracy:.3f}')
        print(f'  validation accuracy: {val_accuracy:.3f}')

        history['train_loss'].append(train_loss);  history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss);      history['val_accuracy'].append(val_accuracy)

        sys.stdout.flush()

    return history


# In[17]:

tstart = time.time()

history = model_fit(device, model, train_loader, test_loader, Loss, Optimizer, num_epochs=NEpochs)

tend = time.time()

print('Training time: {}'.format(tend-tstart))

# ## Plots

# In[18]:


#default_backend = plt.get_backend()
#print(default_backend)


# In[19]:


#matplotlib.use('inline')
###matplotlib.use('Agg')
#
## Accuracy plot
#plt.plot(history['train_accuracy'])
#plt.plot(history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train','val'], loc='upper left')
#plt.show()
##plt.savefig('mnist-cnn-example-pytorch-accuracy.pdf')
#
## Loss plot
#plt.plot(history['train_loss'])
#plt.plot(history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train','val'], loc='upper left')
#plt.show()
##plt.savefig('mnist-cnn-example-pytorch-loss.pdf')


# ## Confusion matrix

# In[20]:


from sklearn.metrics import classification_report,confusion_matrix

val_loss, val_accuracy, eval_labels, eval_preds = model_eval(device, model, test_loader, Loss)

y_test = [y  for x in eval_labels  for y in x]
y_pred = [y  for x in eval_preds   for y in x]

# Plot statistics

target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
y_test_n = np.array([y.cpu() for y in y_test])
print(classification_report(y_test_n, y_pred, target_names=target_names))
print(confusion_matrix(y_test_n, y_pred))


# ## Save model

# In[21]:


#model_file = "model-MNIST-example-pytorch.pth"
#torch.save(model.state_dict(), model_file)


# In[ ]:

