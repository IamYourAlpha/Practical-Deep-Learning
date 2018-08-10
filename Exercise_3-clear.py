
# coding: utf-8

# # Exercise 3 
# 
# ---
# 
# ## Part 1 - Create convolutional layer using `numpy` ##
# 
# Here we will implement forward propagation of one convolutional layer with zero padding using `numpy` library.
# 
# Convolution functions used in this part of exercise:
#     - Zero Padding
#     - Convolve window 
#     - Convolution forward

# First we need to import all necessary libraries.

# In[2]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import ipywidgets

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)


# **Notation**:
#     
# - $n_H$, $n_W$ and $n_C$ denote respectively the height, width and number of channels of a given layer. If you want to reference a specific layer $l$, you can also write $n_H^{[l]}$, $n_W^{[l]}$, $n_C^{[l]}$. 
# - $n_{H_{prev}}$, $n_{W_{prev}}$ and $n_{C_{prev}}$ denote respectively the height, width and number of channels of the previous layer. If referencing a specific layer $l$, this could also be denoted $n_H^{[l-1]}$, $n_W^{[l-1]}$, $n_C^{[l-1]}$. 
# 

# ### Zero-Padding
# 
# Zero-padding adds zeros around the border of an image. The main benefits of padding are the following:
# 
# - It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer. 
# 
# - It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.
# 
# We will [use np.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html) to pad all the images of a batch of examples X with zeros. Note if you want to pad the array "a" of shape $(5,5,5,5,5)$ with `pad = 1` for the 2nd dimension, `pad = 3` for the 4th dimension and `pad = 0` for the rest, you would do:
# ```python
# a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (..,..))
# ```

# In[3]:


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0) ),'constant',constant_values=(0,0))
    
    return X_pad


# In[4]:


np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])
# Expected output: x.shape = (4, 3, 3, 2)
#                  x_pad.shape = (4, 7, 7, 2)
#                  x[1,1] = [[ 0.90085595, -0.68372786]...[-0.26788808  0.53035547]]
#                  x_pad[1,1] = [[ 0.  0.]...[ 0.  0.]]

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])


# ### Single step of convolution 
# 
# We will apply the filter to a single position of the input. This will be used to build a convolutional unit, which: 
# 
# - Takes an input volume 
# - Applies a filter at every position of the input
# - Outputs another volume (usually of different size)
# 
# See lecture slide 23 for detailed explanation.
# 
# In a computer vision application, each value in the matrix on the left corresponds to a single pixel value, and we convolve a 3x3 filter with the image by multiplying its values element-wise with the original matrix, then summing them up and adding a bias. In this first step of the exercise, you will implement a single step of convolution, corresponding to applying a filter to just one of the positions to get a single real-valued output. 
# 
# Later in this notebook, we'll apply this function to multiple positions of the input to implement the full convolutional operation. 
# 

# In[5]:


def conv_single_step(a_slice_prev, K, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    K -- Kernel parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice and K. Do not add the bias yet.
    s = a_slice_prev * K
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = np.float(Z + b)

    return Z


# In[6]:


np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
K = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, K, b)
print("Z =", Z)
# Expected output: -6.99908945068


# ### Convolutional Neural Networks - Forward pass
# 
# In the forward pass, you will take many filters and convolve them on the input. Each 'convolution' gives you a 2D matrix output. We will then stack these outputs to get a 3D volume, see slides 23-30.
# 
# The function below is implemented to convolve the filters W on an input activation A_prev. This function takes as input A_prev, the activations output by the previous layer (for a batch of m inputs), F filters/weights denoted by W, and a bias vector denoted by b, where each filter has its own (single) bias. Finally you also have access to the hyperparameters dictionary which contains the stride and the padding. 
# 
# 
# **Reminder**:
# The formulas relating the output shape of the convolution to the input shape is:
# $$ n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
# $$ n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
# $$ n_C = \text{number of filters used in the convolution}$$
# 
# For this exercise, we won't worry about vectorization, and will just implement everything with for-loops.
# 

# In[7]:


def conv_forward(A_prev, K, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    K -- Kernel, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's shape 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from K's shape 
    (f, f, n_C_prev, n_C) = K.shape
    
    # Retrieve information from "hparameters" 
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor.
    n_H = int((n_H_prev + 2.*pad - f) / stride + 1)
    n_W = int((n_W_prev + 2.*pad - f) / stride + 1)
    
    # Initialize the output volume Z with zeros.
    Z = np.zeros([m, n_H, n_W, n_C])
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):                                  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                      # Select ith training example's padded activation
        for h in range(n_H):                            # loop over vertical axis of the output volume
            for w in range(n_W):                        # loop over horizontal axis of the output volume
                for c in range(n_C):                    # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell).
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    # Convolve the (3D) slice with the correct filter K and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, K[:,:,:,c], b[:,:,:,c])
                                        
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, K, b, hparameters)
    
    return Z, cache


# In[8]:


np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
K = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, K, b, hparameters)
print("Z's mean =", np.mean(Z))
print("Z[3,2,1] =", Z[3,2,1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
# Expectes output: Z's mean = 0.0489952035289
#                  Z[3,2,1] = [-0.61490741, -6.7439236, ... 5.18531798, 8.75898442]
#                  cache_conv[0][1][2][3] = [-0.20075807, 0.18656139, 0.41005165]


# ## Part 2 - Create Convolutional Neural network using PyTorch
# 
# 

# First, we need to import all libraries, that are needed to finish this part of exercise.

# In[9]:


import torch
from torch import nn
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# We need to define default data type and device for Tensors.

# In[10]:


torch.manual_seed(2) # we set up a seed so that your output matches ours although the initialization is random.
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Next, we need to define helper functions to load data. 

# In[11]:


import os
from six.moves.urllib.request import urlretrieve
    
url = 'https://github.com/tejaslodaya/keras-signs-resnet/raw/master/'

def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
          'Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename

def load_dataset():
    if not os.path.exists('datasets/'):
        os.mkdir('datasets')
        
    train_filename = maybe_download('datasets/train_signs.h5', 13281872)
    test_filename = maybe_download('datasets/test_signs.h5', 1477712)
    
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[12]:


# Loading the data (signs)
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()


# The SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.
# 
# The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 

# In[13]:


# Example of a picture
index = 10
plt.imshow(train_set_x_orig[index])
print ("y = " + str(np.squeeze(train_set_y_orig[:, index])))


# In Exercise 2, we had built a fully-connected network for similar cat-non-cat dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.
# 
# To get started, let's examine the shapes of your data. 

# In[14]:


print (train_set_x_orig.shape)
print (train_set_y_orig.shape)
train_set_x = np.reshape(train_set_x_orig,(1080, 3, 64, 64))/255.
test_set_x = np.reshape(test_set_x_orig,(120, 3, 64, 64))/255.
train_set_y = np.reshape(train_set_y_orig,-1).T
test_set_y = np.reshape(test_set_y_orig, -1).T

print ("number of training examples = " + str(train_set_x.shape[0]))
print ("number of test examples = " + str(test_set_x.shape[0]))
print ("train_set_x shape: " + str(train_set_x.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# In this exercise we will train our model with mini-batches to speed up parameter training process.
# 
# First, we will create helper function to split our data into batches. This function will be implemented with `numpy`, and we will retrieve these batches later during training. So, PyTorch Tensors for training will be created after obtaining data with this function.

# In[15]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# As it's stated before, we will train our model with mini-batches. So, train Tensors will be created later. But during test time we don't need mini-batches. Then we can create PyTorch Tensors to use it further to test our Convolutional model.
# 
# **Hint**
# 
# In previous week assignment and exercise Tensor type of input `x` and output `y` was the same, since binary cross-entropy function expects `FloatTensor` as target output `y`. In this week exercise and assignment we will use cross-entropy cost function, because number of classes in the SIGNS dataset is more than 2. This function expects `LongTensor` as target output `y`.

# In[16]:


X_test = torch.tensor(test_set_x, device=device, dtype=dtype,requires_grad=False) 
Y_test = torch.LongTensor(test_set_y, device=device)


# ### Create model
# 
# To specify models that are more complex than a sequence of a few Modules, we can define our own Modules by subclassing `nn.Module` and defining a `forward` which receives input Tensors and produces output Tensors using other modules or other autograd operations on Tensors.
# 
# This class will contain three layers:
# 1. 2D convolutional layer `Conv2d` with `ReLu` activation function, expects kernel size 3,
# 2. 2D convolutional layer `Conv2d` with `ReLu` activation function, expects kernel size 3,
# 3. `Linear` layer without activation function.

# In[17]:


class CNN(nn.Module):
    def __init__(self, image_shape, filters, kernels, strides, n_out):
        super(CNN, self).__init__()
        """
        Arguments:
        image_shape: python list containing shape of input image [channel, height, width]
        filters: python list of integers, defining the number of filters in the Conv layers
        kernels: python list of integers, defining kernel height and width
        strides: python list of integers, defining stride for each Conv layer
        n_out: number of output classes
        
        """
        self.conv1 = nn.Sequential(         # input shape (3, 64, 64)
                    nn.Conv2d(
                        in_channels=image_shape[0], # input channels = 3
                        out_channels=filters[0],    # number of output filters = 8
                        kernel_size=kernels[0],     # kernel size = 3
                        stride=strides[0],          # filter movement/step = 2
                        padding=1,                  # if want same width and length of this image after con2d
                        ),                          # output shape (8, 32, 32)
                    nn.ReLU(),)                      # activation

        self.conv2 = nn.Sequential(         # input shape (8, 32, 32)
                    nn.Conv2d(filters[0], filters[1], 
                              kernels[1], strides[1], 
                              padding=1),     # output shape (16, 16, 16)
                    nn.ReLU(),)                      # activation

        self.out = nn.Linear(16 * 16 * 16, n_out)   # fully connected layer, output 6 classes

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 16 * 16 * 16)
        output = self.out(x)
        return output


# Now we can create our model by passing hyper-parameters to CNN class.

# In[21]:


# Declare hyper-parameters
input_shape = train_set_x.shape
filters = [8, 16]
kernels = [3, 3]
strides = [2, 2]
n_out = int(Y_test.max().item()) + 1
print (n_out)
print (input_shape[1:])
# Define model
model = CNN(input_shape[1:], filters, kernels, strides, n_out)
print(model)
# Expected output: CNN(conv1: Sequential(0:..., 1:...), 
#                      conv2: Sequential(0:..., 1:...), out: Linear...)


# **Loss function**
# 
# Now let's define loss function to train our model. The `nn` package also contains definitions of many popular [loss functions](https://pytorch.org/docs/master/nn.html#loss-functions). In this case we will use Cross-entropy (CE) as our loss function.
# 
# **Optimizer**
# 
# We will update model parameters automatically using `Adam` optimizer. You can check which other optimizers are available in the [documentation](https://pytorch.org/docs/master/optim.html).

# In[18]:


loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# **Define training loop**
# 
# Now everything is ready to define our training loop using `model`, `loss` and `optimizer` created above. We don't need to pass those as parameters of train function, since they are declared as part of PyTorch computational Graph.

# In[19]:


def train(train_x, train_y, X_test, Y_test, minibatch_size=64, num_epochs=20, seed=3):
    """
    Arguments:
    train_x -- numpy.array, training input,
    train_y -- numpy.array, training output target,
    X_test -- Tensor, test input,
    Y_test -- Tensor, test output target,
    minibatch_size -- int, number of training samples per batch,
    num_epochs -- int, number of training epochs,
    seed -- int, seed of random number
    """
    for epoch in range(num_epochs):
        num_minibatches = int(input_shape[0] / minibatch_size)
        seed = seed + 1
        minibatches = random_mini_batches(train_x, train_y, minibatch_size, seed)

        for step, (minibatch_x, minibatch_y) in enumerate(minibatches):
            x = torch.tensor(minibatch_x, device=device, dtype=dtype,requires_grad=False)
            y = torch.LongTensor(minibatch_y, device=device)
            output = model(x)               # cnn output
            loss = loss_func(output, y)     # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients 
                
        print('Epoch: ', epoch+1, '| train loss: %.4f' % loss.data.numpy())
       
                
    test_output = model(X_test)
    train_output = model(torch.tensor(train_x, dtype=dtype))
    pred_test_y = torch.max(test_output, 1)[1].data.squeeze()
    pred_train_y = torch.max(train_output, 1)[1].data.squeeze()
    accuracy_test = float(sum(pred_test_y == Y_test)) / float(Y_test.size(0)) * 100
    accuracy_train = float(sum(pred_train_y.numpy() == train_y)) / float(train_y.shape[0]) * 100
    print('\nTrain accuracy: %.1f\n' % accuracy_train, 'Test accuracy: %.1f' % accuracy_test)


# In[20]:


train(train_set_x, train_set_y, X_test, Y_test)

