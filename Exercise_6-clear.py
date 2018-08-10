
# coding: utf-8

# # Exercise 6. #

# ---

# # Part 1: Create simple autoencoder with PyTorch #
# 
# The aim of this exercise is to create a simple autoencoder (AE). We will train our AE with MNIST database of handwitten digits. 
# 
# ### Introduction ###
# 
# Autoencoders are a specific type of feedforward neural networks where the input is the same as the output. They compress the input into a lower-dimensional code (features) and then reconstruct the output from this representation. The code is a compact “summary” or “compression” of the input, also called the latent-space representation.
# 
# An autoencoder consists of 3 components: encoder, code and decoder. The encoder compresses the input and produces the code, the decoder then reconstructs the input only using this code.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*MMRDQ4g3QvQNc7iJsKM9pg@2x.png" width=600 hight=300 />

# ### AE architecture ###
# 
# Both the encoder and decoder are fully-connected feedforward neural networks. Code is a single layer of an NN with the dimensionality of our choice. The number of nodes in the code layer (code size) is a hyperparameter that we set before training the autoencoder.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png" width=600 hight=300 />
# 
# First the input passes through the encoder, which is a fully-connected ANN, to produce the code. The decoder, which has the similar ANN structure, then produces the output only using the code. The goal is to get an output identical with the input. Note that the decoder architecture is the mirror image of the encoder. This is not a requirement but it’s typically the case. The only requirement is the dimensionality of the input and output needs to be the same. Anything in the middle can be played with.
# 
# There are 4 hyperparameters that we need to set before training an autoencoder:
# 
#  - *Code size*: number of nodes in the middle layer. Smaller size results in more compression.
#  - *Number of layers*: the autoencoder can be as deep as we like. In the figure above we have 2 layers in both the encoder and decoder, without considering the input and output.
#  - *Number of nodes per layer*: the autoencoder architecture we’re working on is called a stacked autoencoder since the layers are stacked one after another. Usually stacked autoencoders look like a “sandwitch”. The number of nodes per layer decreases with each subsequent layer of the encoder, and increases back in the decoder. Also the decoder is symmetric to the encoder in terms of layer structure. As noted above this is not necessary and we have total control over these parameters.
#  - *Loss function*: we either use mean squared error (mse) or binary crossentropy. If the input values are in the range [0, 1] then we typically use crossentropy, otherwise we use the mean squared error. 

# First, we need to import all libraries, that are needed to finish this exercise and do some housekeeping. 

# In[1]:


import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.stats import norm

torch.manual_seed(1) 
sns.set_style('dark')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data preparation ##
# 
# Now we will download the `MNIST` database which consists of handwritten letter images. Next, we eill extract only the training data part and transform them into PyTorch tensor. In addition, each pixel value will be normalized to fit in the range between 0.0 and 1.0

# In[2]:


DOWNLOAD_MNIST = True

# Mnist digits dataset
#train_data = torchvision.datasets.MNIST(
#    root='./mnist/',
#    train=True,                                     # this is training data
#    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
#                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#    download=DOWNLOAD_MNIST,                        # download it if you don't have it
#)
train_data = torchvision.datasets.FashionMNIST(root='./fmnist', train=True,
                                              download=True,
                                              transform=torchvision.transforms.ToTensor(),)


# Let's check the train data size, the corresponding labels size and plot one example.

# In[3]:


# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[1100].numpy(), cmap='gray', interpolation="nearest")
plt.title('%i' % train_data.train_labels[300])
plt.show()


# ### Building the AutoEncoder model ###
# 
# We will build a simple autoencoder using Linear (Dense) neural network layers `tanh` activation dunction. The structure will be as follows:
# 
#  - Encoder
#    - Linear layer, input size = 784 (28*28), output size = 128
#    - `tanh` activation function
#    - Linear layer, input size = 128, output size = 64
#    - `tanh` activation function
#    - Linear layer, input size = 64, output size = 12
#    - `tanh` activation function
#    - Linear layer, input size = 12, output size = 3
#  - Decoder
#    - Linear layer, input size = 3, output size = 12
#    - `tanh` activation function
#    - Linear layer, input size = 12, output size = 64
#    - `tanh` activation function
#    - Linear layer, input size = 64, output size = 128
#    - `tanh` activation function
#    - Linear layer, input size = 128, output size = 784 (28*28)
#    - `sigmoid` activation function
#    
# The ancoder and decoder are simmetric in term of layer sizes. The output of the decoder has `sigmoid` activation function since we want the output values to be between 0.0 and 1.0. We compress the image to only three dimensions because we want to visualize the codes for each digit.

# In[4]:


class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Here, we set the training parameters - epochs, batch size, learning rate, optimizer, loss function, and the number of images to plot at every 100 training steps.

# In[5]:


# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
LR = 0.005         # learning rate
N_TEST_IMG = 5

# Data Loader for easy mini-batch return in training, the image batch shape will be (64, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate the AE
autoencoder = AutoEncoder()

# Use ADAM optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

# The Loss function should be MSE, since we want the AE output to be the same as the input.
loss_func = nn.MSELoss()


# Now, we are ready to train. First, we select several input (N_TEST_IMG) images which will be use to test the AE during training. Then, we run the training loop.

# In[6]:


# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.


for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28*28)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 28*28)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            _, decoded_data = autoencoder(view_data)
            
            # initialize figure
            # plotting decoded image 
            f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
            for i in range(N_TEST_IMG):
                a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray') 
                a[0][i].set_xticks(()); a[0][i].set_yticks(())
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.show()


# After training is finished (you can train with more epochs later), we select first 200 images and pass them through the encoder. The encoder output is 3-dimensional, so we can plot the learned prepresentaiton of each of these 200 images.

# In[ ]:


# visualize in 3D plot
view_data3D = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data3D)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()


# As cen be seen, representations of the different digit images are clustered in different areas of the 3D space!

# ---

# # Part 2: Create variational autoencoder with PyTorch #
# 
# The aim of this exercise is to create a variational autoencoder (VAE). We will train our VAE with the same MNIST database of handwitten digits. 
# 
# In a VAE, there is a strong assumption for the distribution that is learned in the hidden representation. The hidden representation is constrained to be a multivariate guassian. The motivation behind this is that we assume the hidden representation learns high level features and these features follow a very simple form of distribiution. Thus, we assume that each feature is a guassian distribiution and their combination which creates the hidden representation is a multivariate guassian.
# 
#  From a probabilistic graphical models prespective, an auto encoder can be seen as a directed graphical model where the hidden units are latent variables $(z)$ and the following rule applies:
#  
#  $$p_\theta(x, z)=p_\theta(z)p_\theta(x | z)$$
#  
# where $\theta$ indicates that $p$ is parametrized by $\theta$. And according to the Bayes rule, the likelihood of the data $(p_\theta(x))$ can be derived using the following:
# 
# $$p_\theta(x)=\frac{p_\theta(x | z)p_\theta(z)}{p_\theta(x, z)}$$
# 
# $p_\theta(x|z)$ is the distribiution that generates the data $(x)$ and is tractable using the dataset. In a VAE it is assumed the prior distribiution $(p_\theta(z))$ is a multivariate normal distribiution (centered at zero with co-varience of $I$):
# 
# $$p_\theta(z)=\prod_{k=1}^N\cal{N}(z_k|0,1)$$
# 
# 
# The posterior distribiution $(p_\theta(z|x))$ is an intractable distribiution (never observed), but the encoder learns $q_\phi(z|x)$ as its estimator. As mentioned above, we assume $q_\phi(z|x)$ is a normal distribiution which is parameterized by $\phi$:
# 
# $$q_\phi(z|x)=\prod_{k=1}^N\cal{N}(z_k|\mu_k(x), \sigma_k^2(x))$$
# 
# Now the likelihood is parameterized by $\theta$ and $\phi$. The goal is to find a $\theta*$ and a $\phi*$ such that $\log p_{\theta,\phi}(x)$ is maximized. Or equivallently, we minimize the negative log-likelihood (nll). In this setting, the following is a lower-bound on the log-likelihood of $x$:
# 
# $${\cal{L}}(x)=-D_{kl}(q_\phi(z|x)||p_\theta(z))+E_{q_\phi(z|x)}[\log p_\theta(z|x)]$$
# 
# The second term is a reconstruction error which is approximated by sampling from $q_\phi(z|x)$ (the encoder) and then computing $p_\theta(x | z)$ (the decoder). The first term, $D_{kl}$ is the Kullback–Leibler divergence which measures the differnce between two probability distribiutions. The KL term encourges the model to learn a $q_\phi(z|x)$ that is of the form of $p_\theta(z)$ which is a normal distribiution and acts as a regularizer. Considering that $p_\theta(z)$ and $q_\phi(z|x)$ are normal distribiutions, the KL term can be simplified to the following form:
# 
# $$D_{kl}=\frac{1}{2}\sum_{k=1}^N 1 + \log(\sigma_k^2(x)) - \mu_k^2(x) - \sigma_k^2(x)$$
# 
# <img src="https://reyhaneaskari.github.io/Reyhane%20Askari_files/vae.png" >

# ### In short ###
# A variational autoencoder has a very similar structure to an autoencoder except for several changes:
# 
#  - Strong assumption that the hidden representation follows a guassian distribiution.
#  - The loss function has a new regularizer term (KL term) which forces the hidden representation to be close to a normal distribiution.
#  - The model can be used for generation. Since the KL term makes sure that $q_\phi(z|x)$ and $p_\theta(z)$ are close, one can sample from $q_\phi(z|x)$ to generate new datapoints which will look very much like training samples.
# 
# ### The Reparametrization Trick  ###
# The problem that might come to ones mind is that how the gradient flows through a VAE where it involves sampling from $q_\phi(z|x)$ which is a non-deterministic procedure. To tackle this problem, the reparametrization trick is used. In order to have a sample from the distribiution $\cal{N}(\mu, \sigma^2)$, one can first sample from a normal distribiution $\cal{N}(0,1)$ and then calculate:
# 
# $$\cal{N}(\mu, \sigma^2) = \cal{N}(0,1)* \sigma + \mu$$

# ### Implemeting the VAE ###
# 
# Now, we are ready to implement the VAE. It will the folowing structure:
# 
#  - Encoder
#    - Linear layer, 784 inputs, 400 outputs
#    - ReLU activation
#    - Mean (linear) layer, 400 inputs, 20 outputs
#    - LogVar (linear) layer, 400 inputs, 20 outputs
#    
#  - Decoder
#    - Linear layer, 20 inputs, 400 outputs
#    - ReLU Activation
#    - Linear layer, 400 inputs, 784 outputs
#    - Sigmoid activation

# In[7]:


# VAE model
class VAE(nn.Module):
    
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        
        # ENCODER layers
        # 28 x 28 pixels = 784 input pixels, 400 outputs (default)
        self.fc1 = nn.Linear(image_size, h_dim)
        # mu layer, 400 -> 20 latent dimentions (defult)
        self.fc2 = nn.Linear(h_dim, z_dim)
        # logvariance layer,  400 -> 20 latent dimentions (defult)
        self.fc3 = nn.Linear(h_dim, z_dim)
        
        # DECODER layers
        # 20 latent dimentions -> 400 outputs (defult)
        self.fc4 = nn.Linear(z_dim, h_dim)
        # 400 outputs -> 784 (28 x 28) pixels
        self.fc5 = nn.Linear(h_dim, image_size)
        
    # Encoder forward
    def encode(self, x):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected
        21, fully connected 22)

        Parameters
        ----------
        x : [batch_size, 784] matrix; batch_size digits of 28x28 pixels each

        Returns
        -------

        (mu, logvar) : z_dim mean units one for each latent dimension, z_dim
            variance units one for each latent dimension

        """
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    
    # Reparametrization
    def reparameterize(self, mu, log_var):
        """THE REPARAMETERIZATION IDEA:

        For each training sample:

        - take the current learned mu, stddev for each of the z_dim
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [batch_size, z_dim] mean matrix
        logvar : [batch_size, z_dim] variance matrix

        Returns
        -------
        random sample from the learned z-dimensional
        normal distribution;

        """
        # divide log variance by 2, then take exponent
        # yielding the standard deviation
        std = torch.exp(log_var/2)
        # Generate sample from normal distribution with mean 0 and std 1
        eps = torch.randn_like(std)
        # transform the sample as to have mean = mu and std = exp(log_var/2)
        return mu + eps * std

    
    # Decoder forward
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    # Combined forward
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


# The Loss function of the VAE consists of two components:
#  - Reconstruction loss 
#  - KL divergence
#  
# The reconstruction loss shows how well do input x and output recon_x agree. The Kullback–Leibler divergence shows how much does one learned distribution deviate from another, in this specific case the learned distribution from the unit Gaussian.

# In[11]:


def loss_fn(x_reconst, x, mu, log_var):
    
    # Compute reconstruction loss 
    reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
    
    # For KL divergence (see the formula above)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
    # retrun the total loss
    loss = reconst_loss + kl_div
    
    return loss


# Training function follows the general training pattern.

# In[19]:


N_TEST_IMG = 5
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.

def train(model, optimizer, train_loader, epochs):
    
    # List to save losses from each batch
    losses = []

    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(train_loader):
 
            x = x.view(-1, 28*28) 
    
            # Forward pass
            # Get the mu and log_var needed for the loss comutation
            x_reconst, mu, log_var = model(x)
        
            # Compute the loss function
            loss = loss_fn(x_reconst, x, mu, log_var)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
            # Save current loss
            losses.append(loss.item()/x.size(0))
            decoded, _,_ = model(view_data)
        
            if (i+1) % 100 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()/x.size(0)))
                f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
                for i in range(N_TEST_IMG):
                    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray') 
                    a[0][i].set_xticks(()); a[0][i].set_yticks(())
                    a[1][i].imshow(np.reshape(decoded.data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(()); a[1][i].set_yticks(())
                plt.show()
            
            
            
            
    return losses


# Here, we set the VAE parameters - hidden size (h_dim) and latent space size (z_dim), as well as lreaning rate and the number of training epochs. Adam optimizer is used for parameter updates. At the end, we plot the loss.

# In[ ]:


h_dim = 400
z_dim = 20
num_epochs = 2
learning_rate = 1e-3

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = train(model, optimizer, train_loader, num_epochs)

plt.figure(figsize=(10,5))
plt.plot(losses)
plt.show()


# Now, we can check how the model has learned to reconstruct input images. Finction `Visualize` takes several input images (`view_data`), passes them through the VAE and shows both the input and output result.

# In[21]:


def Visualize(model, view_data):

    decoded_data, _, _ = model(view_data)

    # initialize figure
    # plotting decoded image 
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray') 
        a[0][i].set_xticks(()); a[0][i].set_yticks(())
        a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(()); a[1][i].set_yticks(())
    plt.show()
    
Visualize(model, view_data)


# In order to ve able to visualize the latent space, as we did in the case of the simple autoencoder, we  are goint to train a model with 2-dimensional latent space.

# In[ ]:


# Instatntiate VAE with 2D latent space.
model2 = VAE(z_dim=2)
# Create optimizer for the new model
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
# Train
losses2 = train(model2, optimizer2, train_loader, num_epochs)

# and plot losses
plt.figure(figsize=(10,5))
plt.plot(losses2)
plt.show()


# Again, we check how good is the reconstruction using the `Visualize` function. The expectation is that it will not be as goog as before when the latent space was 20-dimensional.

# In[71]:


Visualize(model2, view_data)


# To visualize the representation learned by the VAE, we use the first 500 images, pass them through the encoder, and plot the mean value ($\mu$) of each input image. Different color represent images of different digits. The representations of images of the same digit are supposed to be clusterd together.

# In[72]:


view_data2D = train_data.train_data[:500].view(-1, 28*28).type(torch.FloatTensor)/255.
all_labels = train_data.train_labels[:500].numpy()

z_means, _ = model2.encode(view_data2D)
z_means_x = z_means[:,0].data.numpy()
z_means_y = z_means[:,1].data.numpy()

plt.figure(figsize=(6.5,5))
plt.scatter(z_means_x,z_means_y,c=all_labels,cmap='inferno')
plt.colorbar()
plt.show()


# The VAE ia e generative model, i.e. it can generate images of handwritten digits from the information it has learned during training. To generate an output image, we need to input a particular latent vector `z` to the decoder. In the function `visualize_decoder` below, we generate several `z` vectors linearly distributed (with option range_type='l') or normaly distributed (range_type='g') along the 2D space. Then, we show the generated images from this latent space grid.

# In[73]:


# Visualize digits generated from latent space grid
def visualize_decoder(model, num=20, range_type='g'):
    
    image_grid = np.zeros([num*28, num*28])

    if range_type == 'l': # linear range
        range_space = np.linspace(-4, 4, num)
    elif range_type == 'g': # gaussian range
        range_space = norm.ppf(np.linspace(0.01, 0.99, num))
    else:
        range_space = range_type

    for i, x in enumerate(range_space):
        for j, y in enumerate(reversed(range_space)):
            z = torch.tensor([[x,y]])
            image = model.decode(z)
            image = image.data.numpy()
            image_grid[(j*28):((j+1)*28), (i*28):((i+1)*28)] = image.reshape(28,28)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_grid)
    plt.show()
    
visualize_decoder(model2, 20, 'l')
#visualize_decoder(model2)


# ---

# # Part 3: Create simple GAN with PyTorch #
# 
# The aim of this exercise is to create a simple generative adversarial network (GAN). We will train our VAE with MNIST database of handwitten digits. 
# 
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*5rMmuXmAquGTT-odw-bOpw.jpeg" width=600 hight=300>
# 
# Generative Adversarial Networks are composed of two models:
# 
#  - The first model is called a Generator and it aims to generate new data similar to the expected one. The Generator could be asimilated to a human art forger, which creates fake works of art.
#  - The second model is named the Discriminator. This model’s goal is to recognize if an input data is ‘real’ — belongs to the original dataset — or if it is ‘fake’ — generated by a forger. In this scenario, a Discriminator is analogous to the police (or an art expert), which tries to detect artworks as truthful or fraud.
# 
# How do these models interact? It can be thought of the Generator as having an adversary, the Discriminator. The Generator (forger) needs to learn how to create data in such a way that the Discriminator isn’t able to distinguish it as fake anymore. The competition between these two teams is what improves their knowledge, until the Generator succeeds in creating realistic data.
# 
# A neural network $G(z, \theta_1)$ is used to model the Generator mentioned above. It’s role is mapping input noise variables $z$ to the desired data space $x$ (say images). Conversely, a second neural network $D(x, \theta_2)$ models the discriminator and outputs the probability that the data came from the real dataset, in the range (0,1). In both cases, $\theta_i$ represents the weights or parameters that define each neural network.
# 
# As a result, the Discriminator is trained to correctly classify the input data as either real or fake. This means it’s weights are updated as to maximize the probability that any real data input $x$ is classified as belonging to the real dataset, while minimizing the probability that any fake image is classified as belonging to the real dataset. In more technical terms, the loss/error function used maximizes the function $D(x)$, and it also minimizes $D(G(z))$.
# 
# Furthermore, the Generator is trained to fool the Discriminator by generating data as realistic as possible, which means that the Generator’s weight’s are optimized to maximize the probability that any fake image is classified as belonging to the real datase. Formally this means that the loss/error function used for this network maximizes $D(G(z))$.
# 
# After several steps of training, if the Generator and Discriminator have enough capacity (if the networks can approximate the objective functions), they will reach a point at which both cannot improve anymore. At this point, the generator generates realistic synthetic data, and the discriminator is unable to differentiate between the two types of input.
# 
# Since during training both the Discriminator and Generator are trying to optimize opposite loss functions, they can be thought of two agents playing a minimax game with value function $V(G,D)$. In this minimax game, the generator is trying to maximize it’s probability of having it’s outputs recognized as real, while the discriminator is trying to minimize this same value:
# 
# $$\min_G\max_DV(D,G)=E_{x\sim p_{data}(x)}[logD(x)]+E_{z\sim p_{z}(z)}[log(1-D(G(z)))]$$

# ### Defining Networks ###
# 
# We’ll define the neural networks, starting with the **Discriminator**. This network will take a flattened image as its input, and return the probability of it belonging to the real dataset, or the synthetic dataset. The input size for each image will be 28x28=784. Regarding the structure of this network, it will have two hidden layers, each followed by a Leaky-ReLU nonlinearity. A Sigmoid function is applied to the real-valued output to obtain a value in the open-range (0, 1).
# 
# On the other hand, the **Generative** Network takes a latent variable vector as input, and returns a 784 valued vector, which corresponds to a flattened 28x28 image. Remember that the purpose of this network is to learn how to create undistinguishable images of hand-written digits, which is why its output is itself a new image.
# 
# This network will have two hidden layers, each followed by a ReLU nonlinearity. The output layer will have a `TanH` activation function, which maps the resulting values into the (-1, 1) range, which is the same range in which our preprocessed MNIST images is bounded.

# In[61]:


image_size = 784
latent_size = 64
hidden_size = 256

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())


# The loss function we’ll be using for this task is the Binary Cross Entopy Loss (BCE Loss), and it will be used for this scenario as it resembles the log-loss for both the Generator and Discriminator defined earlier.
# 
# Here, we’ll use `Adam` as the optimization algorithm for both neural networks, with a learning rate of 0.0002. The proposed learning rate was obtained after testing with several values, though it isn’t necessarily the optimal value for this task.

# In[62]:


# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


# This is just an utility function to reset gradients of both the Disctiminator and Generator.

# In[63]:


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# ### Training GAN ###
# 
# The fundamental steps to train a GAN can be described as following:
# 
#  1. Sample a noise set and a real-data set, each with size batch_size.
#  * Train the Discriminator on this data.
#  * Sample a different noise subset with size m.
#  * Train the Generator on this data.
#  * Repeat from Step 1.
#  
# and the training loop will lokk like this:

# In[ ]:


# Set epoch number (try with 200 epochs for better results)
num_epochs = 2

view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
# Start training
total_step = len(train_loader)

for epoch in range(num_epochs):
    
    for i, (images, _) in enumerate(train_loader):
        
        curr_batch_size = images.size()[0]
        images = images.reshape(curr_batch_size, -1)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(curr_batch_size, 1)
        fake_labels = torch.zeros(curr_batch_size, 1)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(curr_batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(curr_batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        
        #Compute the cross-entropy loss with "real" as target (1s). This is what the G wants to do
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
           
            z = torch.randn(curr_batch_size, latent_size)
            fake_images = G(z)
            f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
            for i in range(N_TEST_IMG):
                a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray') 
                a[0][i].set_xticks(()); a[0][i].set_yticks(())
                a[1][i].imshow(np.reshape(fake_images.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.show()
            
            


# After the trainng, we can take a look at some generated images. For this, we just visualize the first 10 fake images from the last trainng step.

# In[ ]:


f, a = plt.subplots(2, 5, figsize=(5, 2))
for i in range(5):
    a[0][i].imshow(np.reshape(fake_images.data.numpy()[i], (28, 28)), cmap='gray') 
    a[0][i].set_xticks(()); a[0][i].set_yticks(())
    a[1][i].imshow(np.reshape(fake_images.data.numpy()[2*i], (28, 28)), cmap='gray')
    a[1][i].set_xticks(()); a[1][i].set_yticks(())
plt.show()


# With only 2 training epochs, generated images all look the same. THat is why many more training epochs are necessary!

# ---
