
# coding: utf-8

# # Exercise 5 
# 
# ---
# 
# ## Part 1 - operations on word vectors ##
# 
# - Load pre-trained word vectors, 
# - Measure similarity using cosine similarity,
# - Use word embeddings to solve word analogy problems such as Man is to Woman as King is to ______.
# 

# There are several algorithms to obtain meaningfull word vectors. `Word2Vec` algorithm was introduced in the last lecture. In this exercise we will use `GloVe` word vectors. `GloVe` also learns vectors of words from their co-occurrence information. They differ in that `Word2Vec` is a "predictive" model, whereas `GloVe` is a "count-based" model. 
# 
# 
# In this part of exercise we will use `numpy` library.

# In[4]:


import numpy as np


# We need to define helper functions to load data. 

# In[7]:


import os
import zipfile
from six.moves.urllib.request import urlretrieve
    
url = 'http://nlp.stanford.edu/data/'

def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists('datasets/'+filename):
        filename, _ = urlretrieve(url + filename, 'datasets/'+filename)
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
        
    pre_train_filename = maybe_download('glove.6B.zip', 862182613)
    with zipfile.ZipFile('datasets/glove.6B.zip') as f:
        data = f.read(f.namelist()[0]).splitlines()
    words = set()
    word_to_vec_map = {}
       
    for line in data:
        line = line.decode().strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map


# Next, lets load the word vectors. Here, we will use 50-dimensional `GloVe` vectors to represent words. Run the following cell to load: 
# - `words`: set of words in the vocabulary.
# - `word_to_vec_map`: dictionary mapping words to their GloVe vector representation.

# In[8]:


words, word_to_vec_map = load_dataset()


# ### Cosine similarity
# 
# Cosine similarity between two word vectors provides an effective method for measuring the degree of linguistic or semantic similarity between two word vectors. 
# 
# Given two vectors $u$ and $v$, cosine similarity is defined as follows: 
# 
# $$\text{CosineSimilarity(u, v)} = \frac {u . v} {||u||_2 ||v||_2} = cos(\theta) \tag{1}$$
# 
# where $u.v$ is the dot product (or inner product) of two vectors, $||u||_2$ is the norm (or length) of the vector $u$, and $\theta$ is the angle between $u$ and $v$. This similarity depends on the angle between $u$ and $v$. If $u$ and $v$ are very similar, their cosine similarity will be close to 1; if they are dissimilar, the cosine similarity will take a smaller value. 

# In[9]:


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    # Compute the dot product between u and v
    dot = np.dot(u, v)
    # Compute the L2 norm of u
    norm_u = np.sqrt(np.sum(u**2))
    
    # Compute the L2 norm of v
    norm_v = np.sqrt(np.sum(v**2))
    # Compute the cosine similarity defined by formula (1)
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity


# In[10]:


father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
k = word_to_vec_map.keys()
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))


# Please feel free to modify the inputs and measure the cosine similarity between other pairs of words! Playing around the cosine similarity of other inputs will give you a better sense of how word vectors behave.

# ### Word analogy task
# 
# In the word analogy task, we complete the sentence <font color='brown'>"*a* is to *b* as *c* is to **____**"</font>. An example is <font color='brown'> '*man* is to *woman* as *king* is to *queen*' </font>. In detail, we are trying to find a word *d*, such that the associated word vectors $e_a, e_b, e_c, e_d$ are related in the following manner: $e_b - e_a \approx e_d - e_c$. We will measure the similarity between $e_b - e_a$ and $e_d - e_c$ using cosine similarity. 

# In[11]:


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    # Get the word embeddings v_a, v_b and v_c
    e_a, e_b, e_c = (word_to_vec_map[x] for x in [word_a, word_b, word_c])
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:        
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c] :
            continue
        
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[w] - e_c))
        
        # If the cosine_sim is more than the max_cosine_sim seen so far,
        # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word


# Run the cell below to perform word analogies, this may take 1-2 minutes.

# In[12]:


triads_to_try = [('italy', 'italian', 'spain'), 
                 ('india', 'delhi', 'japan'), 
                 ('man', 'woman', 'boy'), 
                 ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))


# ## Part 2 - Sequence-to-sequence models
# ---
# In this part of exercise we will build a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25"). 
# As you know fron the last lecture, a Sequence to Sequence network, or seq2seq network, or Encoder Decoder network, is a model consisting of two RNNs called the encoder and decoder. The encoder reads an input sequence and outputs a single vector, and the decoder reads that vector to produce an output sequence. See lecture slides 32~ .

# In[16]:


import torch
import numpy as np

import random
import re
import unicodedata
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# We will need several helper functions to load data.

# In[14]:


get_ipython().system('pip install faker babel')
from faker import Faker
from tqdm import tqdm
from babel.dates import format_date

fake = Faker()
fake.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

LOCALES = ['en_US']

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()
        
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30
    

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], 
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k,v in inv_machine.items()}
 
    return dataset, human, machine, inv_machine

def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    #make lower to standardize
    string = string.lower()
    string = string.replace(',','')
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    return rep

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(nbatch,bsz, data.size(1)).contiguous()
    return data

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty, batch_size=40):
    
    X, Y = zip(*dataset)
    
    X = torch.tensor(np.array([string_to_int(i, Tx, human_vocab) for i in X]), dtype=torch.long, device=device)
    Y = torch.tensor(np.array([string_to_int(t, Ty, machine_vocab) for t in Y]), dtype=torch.long, device=device)
    
    return batchify(X, batch_size), batchify(Y, batch_size)


# In[15]:


dataset, human_vocab, machine_vocab, inv_machine = load_dataset(10000)
print(random.choice(dataset))


# Let's preprocess the data and map the raw text data into the index values. We will also use Tx=30 (which we assume is the maximum length of the human readable date; if we get a longer input, we'd have to truncate it) and Ty=10 (since "YYYY-MM-DD" is 10 characters long).

# In[17]:


Tx = 30
Ty = 10
X, Y = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)


# ## The Encoder
# 
# The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

# In[37]:


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, input.size(0), -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self,batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# ## The Decoder with attention
# 
# The decoder is another RNN that takes the encoder output vector(s) and outputs a sequence of words to create the translation.
# 
# If only the context vector is passed betweeen the encoder and decoder, that single vector carries the burden of encoding the entire sentence.
# 
# Attention allows the decoder network to “focus” on a different part of the encoder’s outputs for every step of the decoder’s own outputs. First we calculate a set of attention weights. These will be multiplied by the encoder output vectors to create a weighted combination. The result (called `attn_applied` in the code) should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.
# 
# Calculating the attention weights is done with another feed-forward layer `attn`, using the decoder’s input and hidden state as inputs. Because there are sentences of all sizes in the training data, to actually create and train this layer we have to choose a maximum sentence length (input length, for encoder outputs) that it can apply to. Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.

# In[254]:


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=Tx):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = torch.nn.Embedding(self.output_size, self.hidden_size)
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.batch_norm = torch.nn.BatchNorm1d(256)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, input.size(0), -1)
        embedded = self.dropout(embedded)
        
        attn_weights = torch.nn.functional.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        output = torch.cat((embedded[0], attn_applied[:,0,:]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.nn.functional.relu(output)
        output = output.reshape(output.size(1),output.size(2))
        output = self.batch_norm(output)
        output = output.reshape(1, output.size(0), output.size(1))
        #print ("size of batch output",output.shape)
        output, hidden = self.gru(output, hidden)
        
        output = torch.nn.functional.log_softmax(self.out(output[0]), dim=1)
        
        return output, hidden, attn_weights

    def initHidden(self,batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# ## Train the model
# 
# First let's make some helper functions to plot losses while training and to print time elapsed and estimated time remaining given the current time and progress %.

# In[255]:


import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))  


# **Train step** 
# First, we write `train()` function to perform one training step over sentence pair.

# In[327]:


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion, max_length=Tx):
    encoder_hidden = encoder.initHidden(40)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
   

    encoder_outputs = torch.zeros(40, max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[:, ei], encoder_hidden)
        encoder_outputs[:,ei,:] = encoder_output

    decoder_input = torch.tensor(np.array([len(machine_vocab)]*40),dtype=torch.long, device=device)
    decoder_input = decoder_input.view(40,-1)

    decoder_hidden = encoder_hidden
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        #decoder_input = target_tensor[:,di]
        
       

        loss += criterion(decoder_output, target_tensor[:,di])


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Epoch training
# 

# In[328]:


def trainIters(encoder, decoder, n_epochs, print_every=1000, plot_every=100, learning_rate=0.02):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    n_iters = n_epochs * X.size(0)

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.9)

    criterion = torch.nn.NLLLoss()
    for ep in range(n_epochs):
        for b_id in range(X.size(0)):
            input_tensor = X[b_id]
            target_tensor = Y[b_id]

            loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if (b_id % print_every == 0) and b_id != 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (b_id+ ep*X.size(0)) / n_iters),
                                         (b_id+ ep*X.size(0)), (b_id+ ep*X.size(0)) / n_iters * 100, print_loss_avg))
                evaluateRandomly(encoder, decoder, 1)
                # Set training mode for encoder and decoder
                encoder.train()
                decoder.train()

            if (b_id % plot_every == 0) and b_id != 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


# ### Evaluation
# 
# Evaluation is mostly the same as training, but there are no targets so we simply feed the decoder’s predictions back to itself for each step. Every time it predicts a word we add it to the output string, and if it predicts the EOS token we stop there. We also store the decoder’s attention outputs for display later.

# In[329]:


def evaluate(encoder, decoder, input_tensor, max_length=Tx):
    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)
        input_length = input_tensor.size(1)
        encoder_hidden = encoder.initHidden(1)

        encoder_outputs = torch.zeros(1, max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[:,ei,:] = encoder_output

        decoder_input = torch.tensor(np.array([len(machine_vocab)]*1),dtype=torch.long, device=device)
        decoder_input = decoder_input.view(1,-1)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(Ty):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(inv_machine[topi.item()])
            
            decoder_input = topi.detach()


        return decoded_words, decoder_attentions[:di + 1]
    
    
def evaluateRandomly(encoder, decoder, n=10):
    # This disables Dropout operation during the test.
    encoder.eval()
    decoder.eval()
    
    for i in range(n):
        pair = random.choice(dataset)
        idx = [dataset.index(pair) // 40, dataset.index(pair) % 40]
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, 
                                            X[dataset.index(pair) // 40, dataset.index(pair) % 40])
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')


# ### Training and Evaluating
# 
# With all these helper functions in place (it looks like extra work, but it makes it easier to run multiple experiments) we can actually initialize a network and start training.
# 
# Remember that the input sentences were heavily filtered. For this small dataset we can use relatively small networks of 256 hidden nodes and a single GRU layer. After about 40 minutes we’ll get some reasonable results.

# In[330]:


hidden_size = 256
encoder1 = EncoderRNN(len(human_vocab), hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, len(machine_vocab)+1, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 3, print_every=50, plot_every=30)


# In[331]:


evaluateRandomly(encoder1, attn_decoder1)


# ---
