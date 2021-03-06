from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import pdb 

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    h1 = X.dot(W1) + b1
    z1 = np.zeros(h1.shape)
    h1 = np.maximum(h1, z1)
    h2 = h1.dot(W2) + b2
    # z2 = np.zeros(h2.shape)
    # h2 = np.maximum(h2, z2)
    scores = h2

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    scores -= np.matrix(np.max(scores, axis=1)).T
    S_exp = np.exp(scores)
    S_exp_sum = np.sum(S_exp, 1)
    loss += np.log(S_exp_sum)
    loss -= scores[np.arange(N), y]
    loss = np.sum(loss)
    loss /=  N 
    loss += reg*(np.sum(W1*W1) + np.sum(W2*W2))
    # Backward pass: compute gradients
    grads = {}

    CW1 = np.zeros(W1.shape)
    Cb1 = np.zeros(b1.shape)
    CW2 = np.zeros(W2.shape)
    Cb2 = np.zeros(b2.shape)

    contrib = np.array(h1 > 0, dtype=float)
    cont_v = np.sum(contrib, 0)
    Cf = S_exp/np.matrix(S_exp_sum).T
    Cf[np.arange(N),y] -= 1

    db2 = np.sum(Cf, 0)
    db2 /= N

    dW2 = h1.T.dot(Cf)
    dW2 /= N
    dW2 += 2*reg*W2

    db1 =np.sum(np.multiply(Cf.dot(W2.T), contrib), 0)
    db1 /= N

    dW1 = np.multiply(W2.dot(Cf.T), contrib.T).dot(X).T
    dW1 /= N
    dW1 += 2*reg*W1
    
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      idx = np.random.choice(num_train, batch_size)
      X_batch = X[idx, :]
      y_batch = y[idx]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      dW1 = grads['W1']
      db1 = np.squeeze(np.array(grads['b1']).T)
      dW2 = grads['W2']
      db2 = np.squeeze(np.array(grads['b2']).T)

      self.params['W1'] -= learning_rate*dW1
      self.params['b1'] -= learning_rate*db1
      self.params['W2'] -= learning_rate*dW2
      self.params['b2'] -= learning_rate*db2

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    scores = self.loss(X)
    y_pred = np.argmax(scores,1)

    return y_pred


