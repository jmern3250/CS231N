from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *



class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim,hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim,1))
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros((num_classes,1))


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']

        h1, h1c = affine_relu_forward(X, w1, b1)
        out, outc = affine_forward(h1, w2, b2)

        scores = out
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, dx = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(w1*w1) + np.sum(w2*w2))

        grads = {}
        dx, dw2, db2 = affine_backward(dx, outc)
        grads['W2'] = self.reg*w2 + dw2
        grads['b2'] = np.reshape(db2, (-1, 1))

        dx, dw1, db1 = affine_relu_backward(dx, h1c)

        grads['W1'] = self.reg*w1 + dw1
        grads['b1'] = np.reshape(db1, (-1,1))

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = len(hidden_dims) +1
        self.dtype = dtype
        self.params = {}

        previous_dim = input_dim
        hidden_dims.append(num_classes)
        for layer, dim in enumerate(hidden_dims):
            self.params['W%s' % str(layer+1)] = np.random.normal(scale=weight_scale, size=(previous_dim, dim)) 
            self.params['b%s' % str(layer+1)] = np.zeros((dim,1))
            previous_dim = dim 
            if self.use_batchnorm and layer+1 < self.num_layers:
                self.params['gamma%s'%str(layer+1)] = np.zeros(dim)+1.0
                self.params['beta%s'%str(layer+1)] = np.zeros(dim)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        out = X 
        cache = []
        for layer in range(self.num_layers):
            w = self.params['W%s'%str(layer+1)]
            b = self.params['b%s'%str(layer+1)]
            if layer+1 < self.num_layers:
                if self.use_batchnorm:
                    gamma = self.params['gamma%s'%str(layer+1)]
                    beta = self.params['beta%s'%str(layer+1)]
                    out, cac = affine_batchnorm_relu_forward(out, gamma, beta, w, b, self.bn_params[layer])
                else:
                    out, cac = affine_relu_forward(out, w, b)
            else: 
                out, cac = affine_forward(out, w, b)
            if self.use_dropout: 
                out, cac_dropout = dropout_forward(out, self.dropout_param)
                cache.append((cac, cac_dropout))
            else:
                cache.append(cac)
        scores = out

        # If test mode return early
        if mode == 'test':
            return scores

        loss, dx = softmax_loss(scores, y)

        reg_loss = 0.0
        for layer in range(self.num_layers):
            w = self.params['W%s'%str(layer+1)]
            reg_loss += np.sum(w*w)
        reg_loss *= self.reg*0.5
        loss += reg_loss

        grads = {}
        for inv_layer in range(self.num_layers):
            layer = self.num_layers - inv_layer
            w = self.params['W%s'%str(layer)]
            b = self.params['b%s'%str(layer)]
            if self.use_dropout:
                cac, cac_dropout = cache[layer-1]
                dx = dropout_backward(dx, cac_dropout)
            else: 
                cac = cache[layer-1]
            if self.use_batchnorm:
                if layer == self.num_layers:
                    dx, dw, db = affine_backward(dx, cac)
                else:
                    dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dx, cac)
                    grads['gamma%s'%str(layer)] = dgamma
                    grads['beta%s'%str(layer)] = dbeta
            else: 
                if layer == self.num_layers:
                    dx, dw, db = affine_backward(dx, cac)
                else:
                    dx, dw, db = affine_relu_backward(dx, cac)
            grads['W%s'%str(layer)] = self.reg*w + dw
            grads['b%s'%str(layer)] = np.reshape(db, (-1, 1))
            
        return loss, grads

def affine_batchnorm_relu_forward(x, gamma, beta, w, b, bn_param):
    """
    Convenience layer that perorms an affine transform followed by 
    batch-normalization, followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta

# def affine_batchnorm_forward(x, gamma, beta, w, b, bn_param):
#     """
#     Convenience layer that perorms an affine transform followed by 
#     batch-normalization, followed by a ReLU

#     Inputs:
#     - x: Input to the affine layer
#     - w, b: Weights for the affine layer

#     Returns a tuple of:
#     - out: Output from the ReLU
#     - cache: Object to give to the backward pass
#     """
#     a, fc_cache = affine_forward(x, w, b)
#     out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
#     cache = (fc_cache, bn_cache)
#     return out, cache

# def affine_batchnorm_backward(dout, cache):
#     """
#     Backward pass for the affine-relu convenience layer
#     """
#     fc_cache, bn_cache = cache
#     dbn, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
#     dx, dw, db = affine_backward(dbn, fc_cache)
#     return dx, dw, db, dgamma, dbeta