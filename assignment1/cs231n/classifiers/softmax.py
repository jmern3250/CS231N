import numpy as np
from random import shuffle
from past.builtins import xrange
import pdb 

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  for i in range(N):
    S = X[i].dot(W)
    S -= np.max(S)
    eS = np.zeros([C,1])
    for j, s in enumerate(S): 
      eS[j] = np.exp(s)
    eS_sum = np.sum(eS)
    for j, s in enumerate(S):
      dW[:,j] += np.exp(s)/eS_sum*X[i]
    ey = np.exp(S[y[i]])
    loss -= np.sum(np.log(ey/eS_sum))
    dW[:, y[i]] -= X[i]

  loss /= N 

  dW /= N

  loss += reg*np.sum(W*W)

  dW += reg*2*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  S = X.dot(W)
  S -= np.matrix(np.max(S, axis=1)).T
    
  loss -= S[np.arange(N), y]
  S_sum = np.sum(np.exp(S), axis=1)
  loss += np.log(S_sum)
  loss = np.sum(loss)
  loss /= N 
  loss += reg*np.sum(W*W)
  
  Cf = np.exp(S)/np.matrix(S_sum).T
  Cf[np.arange(N),y] -= 1
  dW = X.T.dot(Cf)
  dW /= N
  dW += reg*2*W
  return loss, dW