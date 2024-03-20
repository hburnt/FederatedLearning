# Project Name

## Description

Briefly describe the project, its purpose, and its key features. Include any relevant context or background information.

## Table of Contents

- [Motivation](#motivation)
- [The Model](#model)
- [Implimented Algorithm](#algorithm)
- [The Dataset](#dataset)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Motivation

The motivation of this project is to replicate the results found in the paper [Communication-Efficient Learning of Deep Networks from Decentralized data](https://arxiv.org/abs/1602.05629). The results were then compared using different scheduling
protocols. These include random schduling, and a form of age based scheduling.

## The Model

As stated in the paper the model used was a convolutional neural network that had two 5x5 convolutional layers first with a channel size of 32 and then 64, both followed with a 2x2 max pooling, followed by a dense layer of 512 units with ReLU activations. 
  
  ![Model Structure](images/CNNStructure.png)

## Implimented Algorithm
The algorithm used to replicated the results can be described by the following mathmatical expression as stated in the [paper](https://arxiv.org/abs/1602.05629).

**Algorithm 1: FederatedAveraging**

The $K$ clients are indexed by $k$; $B$ is the local minibatch size, $E$ is the number of local epochs, and $\eta$ is the learning rate.

**Server executes:**

1. Initialize $w_0$
2. for each round $t = 1, 2, ...$ do
   - $m \leftarrow \max(C \cdot K, 1)$
   - $S_t \leftarrow$ (random set of $m$ clients)
   - for each client $k \in S_t$ in parallel do
     - $w_{k,t+1} \leftarrow \text{ClientUpdate}(k, w_t)$
   - $m_t \leftarrow \sum_{k \in S_t} n_k$
   - $w_{t+1} \leftarrow \sum_{k \in S_t} \frac{n_k}{m_t} w_{k,t+1}$

**ClientUpdate(k, w):** Run on client $k$

1. $B \leftarrow$ (split $P_k$ into batches of size $B$)
2. for each local epoch $i$ from 1 to $E$ do
   - for batch $b \in B$ do
     - $w \leftarrow w - \eta \nabla f(w; b)$
3. return $w$ to server

## The Dataset

It should be noted that the training data in the MNIST dataset is used only by the clients and the test data is used to evaluate the global model after each round.

### Independent & Identically Distributed Case

The results that were replicated were done using the MNIST dataset in two different cases. As described in the paper the IID case calls for splitting the dataset up by first shuffling the data and then splitting it up evenly to the 100
clients. This gives each client 600 samples to train on. This allows each client to have an equal chance to get data from each class.

This was achieved by the following line of python code:

```python
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
```
### Non-Independent & Identically Distributed Case

In the non IID case each client would only get examples of two numbers so each client will not have examples of the other 8 numbers in the dataset
In order to simulate this in python the following line of code was used to split up the dataset giving each client their own dataloader.

```python
traindata = datasets.MNIST('./data', train=True, download=True, transform=transform)
target_labels = torch.stack([traindata.targets == i for i in range(10)])
target_labels_split = []
for i in range(5):
    target_labels_split += torch.split(torch.where(target_labels[(2 * i):(2 * (i + 1))].sum(0))[0], int(60000 / num_clients))
    
traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in target_labels_split]
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
```
This loops through the sorted dataset which selects groups of numbers [0, 1], [2, 3], [4, 5], [6,7], and [8, 9]
## Results

The model was evealuated in 4 different cases using a mini-batch size (B=10) of 10, and 5 local epochs (E=5). The client size being 100 where 10 clients are selected each round (C=0.1). In the IID case a learning rate of $\eta = 0.215$ was used.
In the non-IID case a learning rate of $\eta = 0.1$ was used as these learning rates were said to be "optimal". The amount of rounds when traning and evaluating the IID and non-IID cases. The results of the IID case match those found in the [reasearch paper](https://arxiv.org/abs/1602.05629).

### IID Loss
![Evaluation Loss](images/IIDEvaluationLosses.png)
### IID Accuracy
![Model Accuracy over Rounds](images/IIDAccuracy.png) 

## Conclusion

## References
