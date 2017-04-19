import numpy as np
from random import shuffle
from past.builtins import xrange
import pdb

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]/num_train
        dW[:, y[i]] -= X[i]/num_train

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # dW += reg*2*np.reshape(np.sum(W, 1), (-1, 1))
  # dW += reg*2*np.sum(W)
  dW += reg*2*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """ 
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  C = W.shape[1]
  N = X.shape[0]
  L = np.zeros([C,N])
  LI = np.ones([C,N])
  for i, label in enumerate(y):
    L[label, i] = 1.0
    LI[label, i] = 0.0 
  LI = LI.transpose()
  S = X.dot(W)
  CS = X.dot(W).dot(L)
  CS = CS*np.eye(CS.shape[0], CS.shape[1])
  CS = np.sum(CS, 1)
  CS = np.reshape(CS, (-1, 1))
  M = S - CS + np.ones((N,C))*LI
  zero_v = np.zeros(M.shape)
  margins = np.maximum(M, zero_v)
  loss = np.sum(margins)
  loss /= N
  dW = np.zeros(W.shape)
  # dW + = X.transpose().dot(contrib)

  one_m = np.ones(W.shape)
  contrib = np.array(margins > 0, dtype=float)
  cont_v = np.reshape(np.sum(contrib, 1), (-1,1))
  contrib[np.arange(N), y] = -cont_v.T
  dW += np.dot(X.T, contrib)
  dW /= N

  dW += reg*2*W

  loss += reg * np.sum(W * W) 
  return loss, dW
