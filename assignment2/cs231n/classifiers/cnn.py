from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        HH = filter_size
        WW = filter_size
        pad = (filter_size - 1) // 2
        stride = 1 #assuming based loss function 
        H_ = int(1 + (H + 2 * pad - HH) / stride)
        W_ = int(1 + (W + 2 * pad - WW) / stride)
        D = int(num_filters*H_*W_/4)
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters,
                                                C, HH, WW))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(D, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        import pdb
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # pdb.set_trace()
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        scores, cache3 = affine_forward(out2, W3, b3)

        if y is None:
            return scores

        grads = {}
        loss, dx = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        dx, dw, db = affine_backward(dx, cache3)
        grads['W3'] = dw
        grads['b3'] = db
        dx, dw, db = affine_relu_backward(dx, cache2)
        grads['W2'] = dw
        grads['b2'] = db
        dx, dw, db = conv_relu_pool_backward(dx, cache1)
        grads['W1'] = dw
        grads['b1'] = db
        
        return loss, grads
