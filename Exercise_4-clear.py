
# coding: utf-8

# # Exercise 4. #

# ---

# # Part 1: Manually create RNN with PyTorch #
# 
# The aim of this exercise is to create manually a simple RNN. We will train our RNN to learn sine function. During training we will be feeding our model with one data point at a time, that is why we need only one input neuron x1, and we want to predict the value at next time step. Our input sequence x consists of 20 data points, and the target sequence is the same as the input sequence but it ‘s shifted by one-time step into the future.

# First, we need to import all libraries, that are needed to finish this exercise and do some housekeeping. 

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import os, glob, zipfile
import unicodedata, string
import random, time, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Needed to download some training data
from six.moves.urllib.request import urlretrieve

# Set the random seed to 0
np.random.seed(0)
torch.manual_seed(0)

# Get PyTorch float type
dtype = torch.float


# ## Data preparation ##
# 
# Now we will generate the training data, where x is an input sequence and y is a target sequence. For this purpose, we create the function `DataGen`:

# In[2]:


def DataGen(seq_length):
    """
    Arguments:
    seq_length - Length of the sine wave (in points)
    
    Returns:
    x - single input sequence (tensor of size [20, 1]) 
    y - single target sequence (tensor of size [20, 1])
    data_time_steps - sequence of sine wave time steps (np_array of size [21,])
    """
    
    # Generate time sequence of seq_length + 1 steps
    data_time_steps = np.linspace(2, 10, seq_length + 1)
    
    # Generate sine wave
    data = np.sin(data_time_steps)
    
    # Reshape as one column vector
    data.resize((seq_length + 1, 1))
    
    # Make input tensor
    x = torch.tensor(data[:-1], dtype=dtype, requires_grad=False)
    
    # Make target tensor (shifted input by one sample)
    y = torch.tensor(data[1:], dtype=dtype, requires_grad=False)
    
    return x, y, data_time_steps
    


# ## Creating the RNN model ##
# 
# Next, we will create our network manually without using the PyTorch nn module. We need to create two weight matrices, w1 of size (input_size, hidden_size) for input to hidden connections, and a w2 matrix of size (hidden_size, output_size) for hidden to output connection. Weights are initialized using a normal distribution with zero mean. Note that the input of the hidden nodes consists of the input data (of size 1) and outputs of the hidden nodes from the previous time step!
# 
# We define forward method, it takes input data, previous hidden state vector as arguments and uses two weights matrices. We will create input vector by concatenating the data with the previous hidden state vector. We perform dot product between the input vector and weight matrix W1, then apply `tanh` function as nonlinearity, which works better with RNNs than sigmoid. Then we perform another dot product between new hidden state vector and weight matrix W2. We want to predict continuous value, so we do not apply any nonlinearity at this stage.
# 
# Note that hidden state vector will be used to populate context neurons at the next time step. That is why we return  hidden state vector along with the output of the network.

# In[3]:


class Model:
    
    def __init__(self, data_size, hidden_size, output_size):
        
        # Set correct input layer size.
        input_size = data_size + hidden_size
        
        # Remember hidden layer size
        self.hidden_size = hidden_size
        
        # Create weight matrices. We dont use biases.
        self.W1 = torch.empty(input_size, hidden_size, dtype=dtype, requires_grad=True)
        self.W2 = torch.empty(hidden_size, output_size, dtype=dtype, requires_grad=True)
        
        # Initialize weight matrices with normaly distributed values: mean = 0, std = 0.4 or 0.3.
        torch.nn.init.normal_(self.W1, 0.0, 0.4)
        torch.nn.init.normal_(self.W2, 0.0, 0.3)
        
    
    def forward(self, data, previous_hidden):
        
        # Concatenate input data with the output from the hidden layer of the previous time step
        input = torch.cat((data, previous_hidden), 1)
        # Calculate hidden state vector
        
        hidden = torch.tanh(input.mm(self.W1))
        # Calculate output
        output = hidden.mm(self.W2)
        
        return output, hidden
        
        
    def UpdateParams(self, lr):
        
        # Update models parameters - W1 and W2 using learning rate 'lr'
        with torch.no_grad():
            self.W1 -= lr * self.W1.grad
            self.W2 -= lr * self.W2.grad
        
            # Clear the gradients
            self.W1.grad.zero_()
            self.W2.grad.zero_()
        


# We also manually define the loss function. Since the RNN outputs are continuous, we use Mean Squared Error (MSE) criterion.

# In[4]:


def loss_fn(pred, target):
    """
    Arguments:
    pred - predicted value (RNN output)
    target - taget value
    
    Returns squared difference between 'pred' and 'target' values
    """
    
    return (pred - target).pow(2).sum()/2


# ## Training ##
# 
# Our training loop will be structured as follows.
# 
# - The outer loop iterates over each epoch. Epoch is defined as one pass of all training data. At the beginning of each epoch, we need to initialize our hidden state vector with zeros.
# 
# - The inner loop runs through each element of the sequence. We run forward method to perform forward pass which returns prediction and previous_hidden state which will be used for next time step. Then we compute Mean Square Error (MSE),  which is a natural choice when we want to predict continuous values.  By running backward() method on the loss we calculating gradients, then we update the weights. We’re supposed to clear the gradients at each iteration by calling zero_() method otherwise gradients will be accumulated. 

# In[5]:


def train(model, x, y, epochs, lr):
    
    # Get the timesteps number
    timesteps = x.size(0)
    
    # Trainig loop
    for i in range(epochs):
        
        # Set total loss to zero
        loss = 0
        
        # Initialize the hidden state vector
        previous_hidden = torch.zeros((1, model.hidden_size), dtype=dtype, requires_grad=True)
        
        # Loop over each timestep
        for j in range(timesteps):
            
            # Get input and target for the current timestep
            input = x[j:(j+1)]
            target = y[j:(j+1)]
            
            # Forward operation
            pred, previous_hidden = model.forward(input, previous_hidden)
            
            # Summ losses for each timestep
            loss += loss_fn(pred, target)
            
        if i % 50 == 0:
             print("Epoch: {} loss {}".format(i, loss.data.numpy()))  
             
        # Calculate gradients
        loss.backward()
            
        # Update parameters
        model.UpdateParams(lr)
                  


# Now, we are ready to train the RNN. First, we define the RNN layer sizes. Then, the number of training epochs, input sequence length and the learning rate. 

# In[6]:


# Set experimental conditions
input_size, hidden_size, output_size = 1, 6, 1
epochs = 300
seq_length = 20
lr = 0.1

# Generate input sequence and terget sequence
x, y, time_steps = DataGen(seq_length)

# Create the RNN
rnn = Model(input_size, hidden_size, output_size)

# Train
train(rnn, x, y, epochs, lr)


# ### Making Predictions ###
# 
# Once our model is trained, we can make predictions, at each step of the sequence we will feed the model with single data point and ask the model to predict one value at the next time step.

# In[7]:


def predict (model, x):

    # Initialize list of predicted values.
    predictions = []
    
    # Initialize the hidden state vector
    previous_hidden = torch.zeros((1, model.hidden_size), dtype=dtype, requires_grad=False)
    
    # Number of timesteps
    timesteps = x.size(0)
    
    # Loop over each input point
    for i in range(timesteps):
        
        # Get current input value
        input = x[i:(i+1)]
        
        # Make Prediction
        pred, previous_hidden = model.forward(input, previous_hidden)
        
        # Save predictions 
        predictions.append(pred.data.numpy().ravel()[0])
        
    return predictions


# In[8]:


# Run RNN prediction
pred = predict (rnn, x)


# Let's plot the true and predicted points.

# In[9]:


# Plot input and predicted points
plt.scatter(time_steps[:-1], x.data.numpy(), s=90, label="Actual")
plt.scatter(time_steps[1:], pred, label="Predicted")
plt.legend()
plt.show()


# As you can see, our model did a pretty good job.

# ### Generating a sine wave with RNN ###
# 
# In the prediction mode, we fed the RNN with the actual input values. Now, we will generate a sine wave by giving the RNN only the first value which we generate randomly. Then, the predicted value will be the next RNN input and so on. This way the RNN can generate a sine wave.

# In[10]:


def Generate(model, input, timesteps = 20):
    
    # Initialize list of generated values
    generated = []
    
    # Initialize the hidden state vector
    previous_hidden = torch.zeros((1, model.hidden_size), dtype=dtype, requires_grad=False)
    
    # Set the first input point
    pred = input
    
    # Loop over the desired sequence length
    for i in range(timesteps):
        
        # Predict value and use it as the next input point
        pred, previous_hidden = model.forward(pred, previous_hidden)
        
        # Save generated points
        generated.append(pred.data.numpy().ravel()[0])
        
    return generated


# Now, we run RNN in generation mode.

# In[11]:


# Make random first input
inp = torch.rand((1,1), dtype=dtype, requires_grad=False)

# Generate sine wave
gen = Generate(rnn, inp)


# Let's plot the generated points.

# In[12]:


# Plot generated points
plt.scatter(time_steps[1:], gen, label="Generated")
plt.legend()
plt.show()


# See that the generated wave is very close to a sine wave.

# ---

# # Part 2: Create RNN based name classification system #
# 
# We will be building and training a basic character-level RNN to classify words. A character-level RNN reads words as a series of characters - outputting a prediction and "hidden state" at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.
# 
# Specifically, we'll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling.

# ### Preparing the data ###
# 
# Data will be download from here <https://download.pytorch.org/tutorial/data.zip> and will be extract to the current directory.
# 
# Included in the data/names directory are 18 text files named as "[Language].txt". Each file contains a bunch of names, one name per line, mostly romanized (but we still need to convert from Unicode to ASCII).
# 
# We'll end up with a dictionary of lists of names per language, {language: [names ...]}. The generic variables "category" and "line" (for language and name in our case) are used for later extensibility.

# In[50]:


# Set the data URL
url = 'https://download.pytorch.org/tutorial/'

# Download a file (if it does not exist)
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

# Prepare all necessary data
def load_dataset():
    # Get data (extract the zip file if not done already)
    if not os.path.exists('data'):
        filename = maybe_download('data.zip', 2882130)
        zip_file = zipfile.ZipFile('data.zip')
        zip_file.extractall()
        zip_file.close()
        os.unlink('data.zip')
        print('Extracted data files in ./data folder')
    else:
        print('Data in ./data folder already available.')
        
    # Get a list of all possible letters
    all_letters = string.ascii_letters + " .,;'"
    # The number of all possible letters
    n_letters = len(all_letters)
    
    # Turn a Unicode string to plain ASCII
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters)
    
    # Dictionary to hold language categories and the names of each category
    category_lines = {}
    # List of all languages (categories)
    all_categories = []
    
    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]

    for filename in glob.glob('data/names/*.txt'):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    
    # Retruns:
    # 1. all_letters    - list of all possible letters
    # 2. n_letters      - the number of all letters
    # 3. category_lines - data dictionary
    # 4. all_categories - list of all categories
    # 5. n_categories   - number of all categories
    return all_letters, n_letters, category_lines, all_categories, n_categories


# In[51]:


# Load the dataset
all_letters, n_letters, category_lines, all_categories, n_categories = load_dataset()


# Now we have category_lines, a dictionary mapping each category (language) to a list of lines (names). We also kept track of all_categories (just a list of languages) and n_categories for later reference.

# In[60]:


# Look at letters, categories and several Italian names.
print(all_letters)
print(all_categories)
print(category_lines['Italian'][:5])
print ("The number of categories are %d"%(n_categories))


# ### Turning Names into Tensors ###
# 
# Now that we have all the names organized, we need to turn them into Tensors to make any use of them.
# 
# To represent a single letter, we use a "one-hot vector" of size <1 x n_letters>. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
# 
# To make a word we join a bunch of those into a 2D matrix <line_length x 1 x n_letters>.
# 
# That extra 1 dimension is because PyTorch assumes everything is in batches - we're just using a batch size of 1 here.

# In[16]:


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())


# ### Creating the Network ###
# 
# Before autograd, creating a recurrent neural network in Torch involved cloning the parameters of a layer over several timesteps. The layers held hidden state and gradients which are now entirely handled by the graph itself. This means you can implement a RNN in a very "pure" way, as regular feed-forward layers.
# 
# This RNN module is just 2 linear layers which operate on an input and hidden state, with a LogSoftmax layer after the output.
# 
# <img src="https://i.imgur.com/Z2xbySO.png" alt="" title="Title text" />

# In[17]:


# Define the RNN
class RNN(nn.Module):
    # Initialize the RNN with each layer size
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # Remenber the hidden size
        self.hidden_size = hidden_size

        # Create two layers: input-to-hidden and input-to-output
        # Note that both the layers take the data and (previous) hidden output as input
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # since we do classification we use SoftMax as output layer activation
        self.softmax = nn.LogSoftmax(dim=1)

    # Forward processing
    def forward(self, input, hidden):
        
        # Combine the input data and the output from the previous hidden layer
        combined = torch.cat((input, hidden), 1)
        # Calculate current hidden output
        hidden = self.i2h(combined)
        # Calculate current output
        output = self.i2o(combined)
        output = self.softmax(output)
        
        return output, hidden

    # Initialize the hidden layer
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# Now, we create the RNN with a hidden layer of 64 neurons/cells. 

# In[18]:


n_hidden = 64
rnn = RNN(n_letters, n_hidden, n_categories)


# To run a step of this network we need to pass an input (in our case, the Tensor for the current letter) and a previous hidden state (which we initialize as zeros at first). We'll get back the output (probability of each language) and a next hidden state (which we keep for the next step).

# In[62]:


input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)


# To run a step of this network we need to pass an input (in our case, the Tensor for the current letter) and a previous hidden state (which we initialize as zeros at first). We'll get back the output (probability of each language) and a next hidden state (which we keep for the next step).

# In[41]:


input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[4], hidden)
print(output)


# As you can see the output is a <1 x n_categories> Tensor, where every item is the likelihood of that category (higher is more likely).

# ## Training ##
# 
# ### Preparing for Training ###
# 
# Before going into training we should make a few helper functions. The first is to interpret the output of the network, which we know to be a likelihood of each category. We can use Tensor.topk to get the index of the greatest value:

# In[44]:


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))


# We will also want a quick way to get a training example (a name and its language):

# In[46]:


# Returns random element of list 'l'
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Returns random training sample (name:language pair) in normal and tensor formats
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# Check some random pairs
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)


# ### Training the Network ###
# 
# Now all it takes to train this network is show it a bunch of examples, have it make guesses, and tell it if it's wrong.
# 
# For the loss function nn.NLLLoss is appropriate, since the last layer of the RNN is nn.LogSoftmax.

# In[23]:


loss_fn = nn.NLLLoss()


# Each loop of training will:
# 
#  - Create input and target tensors
#  - Create a zeroed initial hidden state
#  - Read each letter in and
#    - Keep hidden state for next letter
#  - Compare final output to target
#  - Back-propagate
#  - Return the output and loss

# In[24]:


learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    
    # Zero the hidden state
    hidden = rnn.initHidden()
    
    # Remove gradients
    rnn.zero_grad()

    # Loop over the input sequence
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # Calculate loss
    loss = loss_fn(output, category_tensor)
    
    # Calculate gradiants
    loss.backward()

    # Update RNN parametrs
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    # Return the RNN output and loss
    return output, loss.item()


# Now we just have to run that with a bunch of examples. Since the train function returns both the output and loss we can print its guesses and also keep track of loss for plotting. Since there are 1000s of examples we print only every print_every examples, and take an average of the loss.

# In[25]:


n_iters = 10000
print_every = 200
plot_every = 100

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), current_loss / print_every, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# ### Plotting the Results ###
# Plotting the historical loss from all_losses shows the network learning:

# In[26]:


plt.figure()
plt.plot(all_losses)


# ### Evaluating the Results ###
# 
# To see how well the network performs on different categories, we will create a confusion matrix, indicating for every actual language (rows) which language the network guesses (columns). To calculate the confusion matrix a bunch of samples are run through the network with evaluate(), which is the same as train() minus the backprop.

# In[27]:


# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
# Number od tests
n_confusion = 2000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()


# ---
