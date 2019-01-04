# third party
import tensorflow as tf


class Flatten():
    ''' Flatten layer used to reshape inputs into vector representation
    
    Layer should be used in the forward pass before a dense layer to 
    transform a given tensor into a vector. 
    '''
    def __init__(self):
        self.params = []

    def forward(self, X):
        ''' Reshapes a n-dim representation into a vector 
            by preserving the number of input rows.
        
        Args:
            X: Images set
    
        Returns:
            X_: Matrix with images in a flatten represenation
            
        Examples:
            [10000, 1, 28, 28] -> [10000,784]
        '''
        return tf.reshape(X, [-1, X.shape[1] * X.shape[2] * X.shape[3]])
    

class FullyConnected():
    ''' Fully connected layer implemtenting linear function hypothesis 
        in the forward pass and its derivation in the backward pass.
    '''
    def __init__(self, in_size, out_size, activation_func=None,stddev=0.1):
        ''' Initilize all learning parameters in the layer
        
        Weights will be initilized with modified Xavier initialization.
        Biases will be initilized with zero. 
        '''
        self.W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=stddev))
        self.b = tf.Variable(tf.ones([out_size])/10)
        self.params = [self.W, self.b]
        self.activation_func = activation_func
        self.out_size = out_size

    def forward(self, X):
        ''' Linear combiationn of images, weights and bias terms
            
        Args:
            X: Matrix of images (flatten represenation)
    
        Returns:
            out: Sum of X*W+b  
        '''
        Z = tf.matmul(X, self.W) + self.b
        if self.activation_func is None:
            return Z
        else:
            return self.activation_func.forward(Z)
        
        
class Convolution():
    ''' Fully connected layer implemtenting linear function hypothesis 
        in the forward pass and its derivation in the backward pass.
    '''

    def __init__(self, input_channels=1, filter_num=32, filter_dim=(3, 3), stride=1, activation_func=None):
        self.W = tf.Variable(tf.truncated_normal([filter_dim[0], filter_dim[1], input_channels, filter_num], stddev=0.1))
        self.b = tf.Variable(tf.ones([filter_num])/10)
        self.params = [self.W, self.b]
        self.stride = stride
        self.activation_func = activation_func

    def forward(self, X):
        ''' Linear combiationn of images, weights and bias terms

        Args:
            X: Matrix of images (flatten represenation)

        Returns:
            out: Sum of X*W+b  
        '''
        Z = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding='SAME') + self.b
        if self.activation_func is None:
            return Z
        else:
            return self.activation_func.forward(Z)