
# coding: utf-8

# # Exercise 2. #

# ---

# 
# # Part 1: Create MLP using NumPy #
# 
# The aim of this exercise is to create simple neural network using NumPy library

# First, we need to import all libraries, that are needed to finish this exercise.

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage
import os
from six.moves.urllib.request import urlretrieve

get_ipython().run_line_magic('matplotlib', 'inline')


# Next, we need define some helper functions to load data. 

# In[2]:


url = 'https://github.com/andersy005/deep-learning-specialization-coursera/raw/master/01-Neural-Networks-and-Deep-Learning/week2/Programming-Assignments/'

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
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    train_filename = maybe_download('datasets/train_catvnoncat.h5', 2572022)
    test_filename = maybe_download('datasets/test_catvnoncat.h5', 616958)

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In this exercise, we will use a simple image dataset that has two classes *cat* and *non-cat*. Every image is represented with numpy array of shape \[num_pixels, num_pixels, 3\] (3 is number of RGB channels). 

# In[3]:


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Figuring out the dimensions and shapes of the problem
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# You can see images from dataset by changing index:

# In[4]:


# Example of a picture
index = 1
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


# As you know from the lecture, standard neural network expects 1D vector as input. So, we need to reshape these images in a numpy-array of shape (num_pixels $*$ num_pixels $*$ 3, 1).
# 
# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use: 
# ```python
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
# ```

# In[5]:


# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.

# In[6]:


# "Standardize" the data
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# ## Building basic parts of Neural Network ##
# 
# **Mathematical expression of the neural network classifier with one linear hidden layer**:
# 
# For one example $x^{(i)}$:
# $$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
# $$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
# $$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$
# 
# The cost is then computed by summing over all training examples:
# $$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{4}$$
# 

# **The general methodology to build a Neural Network is to:**
# 1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
# 2. Initialize the model's parameters
# 3. Training loop:
#     - Implement forward propagation
#     - Compute loss
#     - Implement backward propagation to get the gradients
#     - Update parameters (gradient descent)
# 
# We will start with building helper functions to compute steps 1-3 and then merge them into one function we call `model()`. Once we've built `model()` and learnt the right parameters, we can make predictions on new data.

# ### Step 1: Defining structure of the neural network ###
# 
# We will use shapes of input and output to define sizes of input and output layers and set size of hidden layer to 4.

# In[7]:


# Step 1:
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
   
    return (n_x, n_h, n_y)


# In[8]:


# Let's check if we implemented previous function correctly
X_tmp = np.zeros([10,50])
Y_tmp = np.zeros([1,50])
n_x, n_h, n_y = layer_sizes(X_tmp, Y_tmp)
print("The size of the input layer is: n_x = " + str(n_x))  # Expected output: 10
print("The size of the hidden layer is: n_h = " + str(n_h)) # Expected output: 4
print("The size of the output layer is: n_y = " + str(n_y)) # Expected output: 1


# ### Step2: Initialize parameters of the model ###
# 
# Given sizes of input, hidden and output layer we can create and initialize:
# - weight matrices with random values.
# - biases with zeros.

# In[9]:


# Step 2:
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    W1 = np.random.randn(n_h,n_x) * 0.01 
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01 
    b2 = np.zeros((n_y,1)) 
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[10]:


# Let's check implementation
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"])) # [[-4.16757847e-03, ...], ... ,[... , 5.42352572e-03]]
print("b1 = " + str(parameters["b1"])) # [[0.], [0.], [0.], [0.]]
print("W2 = " + str(parameters["W2"])) # [[-0.00313508  0.00771012 -0.01868091  0.01731185]]
print("b2 = " + str(parameters["b2"])) # [[0.]]


# ### Step 3: Training loop ###
# 
# First we need to implemente `sigmoid()` function.

# In[11]:


# Helper function
def sigmoid(x):
    """ 
    Argument:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """

    s = 1/(1+np.exp(-x)) 
    
    return s


# In[12]:


# Let's check implementation
s = sigmoid(0)
print('sigmoid(0) = ' + str(s)) # Expected output: 0.5


# **Step 3.1:**
#     Implement forward propagation function
#     
# - Look above at the mathematical representation of our classifier.
# - We will use the function `np.tanh()` for hidden layer and function `sigmoid()` for output layer.
# - The steps we need to implement are:
# 1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
# 2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
# 
# Values needed in the backpropagation are stored in "`cache`". The `cache` will be given as an input to the backpropagation function.

# In[13]:


# Step 3.1:
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.add(np.dot(W1,X), b1)
    A1 = np.tanh(Z1)
    Z2 = np.add(np.dot(W2,A1), b2)
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# In[14]:


# Let's check implementation
A2, cache = forward_propagation(X_tmp, parameters)

print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
# Expected output: 0.0 0.0 0.0 0.5


# **Step 3.2:** Implement `compute_cost()` to compute the value of the cost $J$.
# 
# Now that we have computed $A^{[2]}$ (in the Python variable "`A2`"), which contains $a^{[2](i)}$ for every example, we can compute the cost function as follows:
# 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{5}$$
# 

# In[15]:


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (5)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- binary cross-entropy cost given equation (5)
    """
    
    m = Y.shape[1] # number of examples

    # Compute the cross-entropy cost
    logprobs = np.multiply(-np.log(A2),Y) + np.multiply(-np.log(1 - A2), 1 - Y)
    cost = (1./m) * np.nansum(logprobs)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost


# In[16]:


# Let's check implementation
print("cost = " + str(compute_cost(A2, Y_tmp, parameters)))
# Expected output: 0.69314718


# **Step 3.3:** Implement backward_propagation function
# 
# Using the cache computed during forward propagation, we can now implement `backward_propagation()`.
# 
# Backpropagation is usually the hardest (most mathematical) part in deep learning. Here are equations in the notation we used in this notebook.  
# 
# $$\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } = a^{[2](i)} - y^{(i)}\tag{6}$$
# 
# $$\frac{\partial \mathcal{J} }{ \partial W_2 } = \frac{1}{m}  \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } a^{[1] (i) T}\tag{7} $$
# 
# $$\frac{\partial \mathcal{J} }{ \partial b_2 } = \frac{1}{m}  \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)}}}\tag{8}$$
# 
# $$\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} } =  W_2^T \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2})\tag{9} $$
# 
# $$\frac{\partial \mathcal{J} }{ \partial W_1 } = \frac{1}{m}  \frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} }  X^T \tag{10} $$
# 
# $$\frac{\partial \mathcal{J} _i }{ \partial b_1 } = \frac{1}{m}  \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)}}}\tag{11}$$
# 
# - Note that $*$ denotes elementwise multiplication.
# 

# In[17]:


# Step 3.3
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"] 
    A2 = cache["A2"] 
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y # equation (6)
    dW2 = (1./m) * (np.dot(dZ2, A1.T)) # equation (7)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True) # equation (8)
    dZ1 = (W2.T * dZ2) * (1 - np.power(A1,2)) # equation (9)
    dW1 = (1./m) * np.dot(dZ1, X.T) # equation (10)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True) # equation (11)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# In[18]:


# Let's check implementation
grads = backward_propagation(parameters, cache, X_tmp, Y_tmp)
print ("dW1 = "+ str(grads["dW1"])) # Expected output: [[0., 0., ...], ...,[..., 0., 0.]]
print ("db1 = "+ str(grads["db1"])) # Expected output: [[-0.00156754], [ 0.00385506], [-0.00934045], [ 0.00865592]]
print ("dW2 = "+ str(grads["dW2"])) # Expected output: [[0. 0. 0. 0.]]
print ("db2 = "+ str(grads["db2"])) # Expected output: [[0.5]]


# **Step 3.4:** Implement the update rule. 
# 
# We will use gradient descent as the update rule. We have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2). We can retrieve those from dictionaries `parameters` and `grads`.
# 
# **General gradient descent rule**: 
# $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.

# In[19]:


# Step 3.4
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
        
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[20]:


# Let's check implementation
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"])) # Expected output: [[-4.16757847e-03, ...], ... ,[...,5.42352572e-03]]
print("b1 = " + str(parameters["b1"])) # Expected output: [[ 0.00188105], [-0.00462607], [ 0.01120854], [-0.01038711]]
print("W2 = " + str(parameters["W2"])) # Expected output: [[-0.00313508  0.00771012 -0.01868091  0.01731185]]
print("b2 = " + str(parameters["b2"])) # Expected output: [[-0.6]]


# ### Predictions ###
# 
# Now we can calculate prediction probability of example to belong to class cat.
# 
# We can use parameters from the `forward_propagation()` to get those probabilities. By comparing them with threshold we can get predicted class (non-cat: 0 / cat: 1).
# 
# $$ predictions = $y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
#       1 & \text{if}\ activation > 0.5 \\
#       0 & \text{otherwise}
#     \end{cases}$  $$
#     

# In[21]:


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (non-cat: 0 / cat: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    
    return predictions


# ### Integrate Step 1, Step 2 and Step 3 and build full model###
# 
# Now we can build our neural network model in `model()` function.
# 

# In[22]:


def model(X_train, Y_train, X_test, Y_test, n_h, num_iterations=10000, 
          learning_rate=0.5, print_cost=False):
    """
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    n_h -- size of the hidden layer
    num_iterations -- number of iterations in gradient descent loop
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 200 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X_train, Y_train)[0]
    n_y = layer_sizes(X_train, Y_train)[2]
    costs = []
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X_train, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y_train, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X_train, Y_train)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate) 
        
        # Print the cost every 200 iterations
        if i % 200 == 0:
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    # Predict test/train set examples
    Y_prediction_test = predict(parameters, X_test)
    Y_prediction_train = predict(parameters, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "costs" : costs,
         "parameters" : parameters,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[23]:


d = model(train_set_x, train_set_y, test_set_x, test_set_y, 10, 
                  num_iterations=200, learning_rate=0.05, print_cost=True)
# Expected output: Cost after iteration 0:  0.693045 ...


# ### Choice of learning rate ###
# 
# In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate $\alpha$  determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.
# 
# 
# - Compare the learning curve of your model with several choices of learning rates. Try your own values.
# 
# **Optional**
# 
# - Find optimal learning rate and hidden layer size for this model.

# In[24]:


learning_rates = [0.01, 0.001, 0.0001] 
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, 10, 
                           num_iterations=2000, learning_rate=i)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# ---

# # Part 2: Create MLP using PyTorch Tensor and Autograd modules. #
# 
# The aim of this part is to create similar structure as in part 1 using PyTorch library

# We need to define default data type and device for Tensors.

# In[25]:


torch.manual_seed(2) # we set up a seed so that your output matches ours although the initialization is random.
dtype = torch.float


# We already load our data in the part 1, but here we need to reshape it in different order. Now we will reshape these images in a numpy-array of shape (num_samples, num_pixels  $*$ num_pixels $*$  3) and labels in a numpy array of shape (num_samples, num_labels).

# In[26]:


# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)
print (train_set_x_flatten[0])
# Reshape labels dataset

train_set_y = train_set_y.T
test_set_y = test_set_y.T 

# "Standardize" the data
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# Then we can create PyTorch Tensors to use it further to train and test our PyTorch model.
# 
# **Hint**
# Here we use previously defined `device` and `dtype`. And we set `requires_grad` parameter to False since it's only data holder.

# In[27]:


X_train = torch.tensor(train_set_x, dtype=dtype,requires_grad=False)
Y_train = torch.tensor(train_set_y, dtype=dtype,requires_grad=False)
X_test = torch.tensor(test_set_x, dtype=dtype,requires_grad=False) 
Y_test = torch.tensor(test_set_y, dtype=dtype,requires_grad=False) 


# In this part of exercise we will follow the same metodology as in **part 1**.
# 
# **The general methodology to build a Neural Network is to:**
# 1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
# 2. Initialize the model's parameters
# 3. Training loop:
#     - Implement forward propagation
#     - Compute loss
#     - Implement backward propagation to get the gradients
#     - Update parameters (gradient descent)

# ### Step 1: Defining structure of the neural network ###
# 
# First, we implement function `layer_sizes()`. We will use sizes of input and output Tensors to define sizes of input and output layers.

# In[28]:


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input tensor of shape (input size, number of examples)
    Y -- labels tensor of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x = X.size(1)
    n_y = Y.size(1)
    
    return n_x, n_y


# In[29]:


# Let's check implemention
X_tmp = torch.zeros([50,10])
Y_tmp = torch.zeros([50,1])
n_x, n_y = layer_sizes(X_tmp, Y_tmp)
print("The size of the input layer is: n_x = " + str(n_x))  # Expected output: 10
print("The size of the output layer is: n_y = " + str(n_y)) # Expected output: 1


# ### Step2: Initialize parameters of the model ###
# Given sizes of input, hidden and output layer we can create and initialize:
# - weight matrices with random values.
# - biases with zeros.
# 
# **Hint** Here we set `requires_grad` parameter to `True`. This will allow us to compute gradients automatically later.

# In[30]:


# Step 2:
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_x, n_h)
                    b1 -- bias vector of shape (1, n_h)
                    W2 -- weight matrix of shape (n_h, n_y)
                    b2 -- bias vector of shape (1, n_y)
    """

    # Create empty tensors
    W1 = torch.empty(n_x, n_h, dtype=dtype, requires_grad=True)
    b1 = torch.empty((1,n_h), dtype=dtype, requires_grad=True) 
    W2 = torch.empty(n_h, n_y, dtype=dtype, requires_grad=True)
    b2 = torch.zeros((1,n_y), dtype=dtype, requires_grad=True) 

    # Initialize tensors
    torch.nn.init.orthogonal_(W1)
    torch.nn.init.constant_(b1, 0)
    torch.nn.init.orthogonal_(W2)
    torch.nn.init.constant_(b2, 0)
    
    assert (W1.size() == (n_x, n_h))
    assert (b1.size() == (1, n_h))
    assert (W2.size() == (n_h, n_y))
    assert (b2.size() == (1, n_y))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[31]:


# Let's check implementation
parameters = initialize_parameters(n_x, 5, n_y)
print("W1 = " + str(parameters["W1"])) # Expected output: tensor([[ 0.1203,...],...,[...,0.0954]])
print("b1 = " + str(parameters["b1"])) # Expected output: tensor([[ 0.,  0.,  0.,  0.,  0.]])
print("W2 = " + str(parameters["W2"])) # Expected output: tensor([[ 0.3309],[ 0.3690],[ 0.1238],[-0.1911],[-0.8381]])
print("b2 = " + str(parameters["b2"])) # Expected output: tensor([[ 0.]])


# ### Step 3: Training loop ###
# **Step 3.1:**
#     Implement forward propagation function
#     
# - Look above at the mathematical representation of our classifier.
# - We will use the function `torch.tanh()` for hidden layer and function `torch.sigmoid()` for output layer.
# - The steps we need to implement are:
# 1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
# 2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
# 
# When we implemented this step in **part 1** we stored values needed in the backpropagation in "`cache`". Here, the forward pass of our network will define a computational graph. So nodes in the graph will be Tensors, and edges will be functions that produce output Tensors from input Tensors. Later we will use [autograd](https://pytorch.org/docs/master/autograd.html) package to calculate backward pass. Since all functions are stored in the computational graph, there's no need to keep them in separate dictionary.

# In[32]:


# Step 3.1:
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (m, n_x)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    # Replace None with your own code
    Z1 = torch.add(torch.mm(X,W1), b1)
    A1 = torch.tanh(Z1)
    Z2 = torch.add(torch.mm(A1, W2), b2)
    A2 = torch.sigmoid(Z2)
    
    assert(A2.size() == (X.size(0), 1))
    
    return A2


# In[33]:


# Let's check implementation

A2 = forward_propagation(X_tmp, parameters)

print(torch.mean(A2).item())
# Expected output: 0.5


# **Step 3.2:** Implement `compute_cost()` to manually compute the value of the cost $J$. 
# 
# Now that we have computed $A^{[2]}$ (in the Python variable "`A2`"), which contains $a^{[2](i)}$ for every example, we can compute the cost function as follows:
# 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{5}$$
# 
# We will implement it in the same way as in **part1**, but now we will use PyTorch functions.
# 

# In[34]:


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (5)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (number of examples, 1)
    Y -- "true" labels vector of shape (number of examples, 1)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- binary cross-entropy cost given equation (5)
    """
    
    m = Y.size(0) # number of example

    # Compute the cross-entropy cost
    logprobs = (-torch.log(A2) * Y) + (-torch.log(1 - A2) * (1 - Y))
    cost = (1./m) * torch.sum(logprobs) 
    
    cost = torch.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    
    return cost


# In[35]:


# Let's check implementation

print("cost = " + str(compute_cost(A2, Y_tmp, parameters).item()))
# Expected output: 0.693147


# **Step 3.3:** Backward propagation
# 
# Here we will use automatic differentiation to automate the computation of backward passes in neural networks. This call will compute the gradient of loss with respect to all Tensors with requires_grad=True. So we will call it later directly in the training loop.
# 
# The `autograd` package in PyTorch provides exactly this functionality. When using `autograd`, the forward pass of your network will define a computational graph. Backpropagating through this graph then allows you to easily compute gradients. See documentation for more details: https://pytorch.org/docs/master/autograd.html
# 

# **Step 3.4:** 
# Implement the update rule using gradient descent. 
# 
# **General gradient descent rule**: 
# 
# $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.
# 
# **Hint** 
# We can access gradients using `parameter.grad` that is automatically computed in the backward pass.
# After updating parameters, computed gradients should be zeroed. We will use `.zero_()` for in-place operation.
#     

# In[36]:


def update_parameters(parameters, learning_rate):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Since we use manual update of parameters, we need to wrap in torch.no_grad()
    # because all parameters have requires_grad=True, but we don't need to track this.
    with torch.no_grad():
        
        # Update each parameter
        parameters["W1"] -= learning_rate * parameters["W1"].grad
        parameters["b1"] -= learning_rate * parameters["b1"].grad
        parameters["W2"] -= learning_rate * parameters["W2"].grad
        parameters["b2"] -= learning_rate * parameters["b2"].grad
        
        # Manually zero the gradients after updating weights
        parameters["W1"].grad.zero_()
        parameters["b1"].grad.zero_()
        parameters["W2"].grad.zero_()
        parameters["b2"].grad.zero_()
        
    return parameters


# ### Predictions
# 
# Now we can calculate prediction probability of example to belong to class cat.
# 
# We can use parameters from the `forward_propagation()` to get those probabilities. By comparing them with threshold we can get predicted class (non-cat: 0 / cat: 1).
# 
# $$ predictions = y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
#       1 & \text{if}\ activation > 0.5 \\
#       0 & \text{otherwise}
#     \end{cases}  $$
# 
# As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: ```X_new = (X > threshold)```. Remember that it will return `LongTensor`, so we need to cast it back to `FloatTensor`.

# In[37]:


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, n_x)
    
    Returns
    predictions -- vector of predictions of our model (non-cat: 0 / cat: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2 = forward_propagation(X, parameters)
    predictions = (A2 > 0.5).float()
    
    return predictions


# ### Integrate Step 1, Step 2 and Step 3 and build full model###
# 
# Now we can build our neural network model in `model()` function.

# In[38]:


def model(X_train, Y_train, X_test, Y_test, n_h, num_iterations=10000, 
          learning_rate=0.5, print_cost=False):
    """
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    n_h -- size of the hidden layer
    num_iterations -- number of iterations in gradient descent loop
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- if True, print the cost every 200 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    n_x, n_y = layer_sizes(X_train, Y_train)
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations): 
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2 = forward_propagation(X_train, parameters) 
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y_train, parameters) 
        print (cost)
        # Backpropagation. Compute gradient of the cost function with respect to all the 
        # learnable parameters of the model. Use autograd to compute the backward pass.
        cost.backward() 
 
        # Gradient descent parameter update.
        parameters = update_parameters(parameters, learning_rate) 
        
        # Print the cost every 50 iterations
        if print_cost and i % 50 == 0:
            print ("Cost after iteration %i: %f" %(i, cost.item()))
            
    # Predict test/train set examples
    Y_prediction_test = predict(parameters, X_test) 
    Y_prediction_train = predict(parameters, X_train) 

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - torch.mean(torch.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - torch.mean(torch.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "parameters" : parameters,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[39]:


d = model(X_train, Y_train, X_test, Y_test, 15, 
                  num_iterations=200, learning_rate=0.05, print_cost=True)


# ## Optional ##
# 
# - Try different initializations in `initialize_parameters()` and find most suitable one.
