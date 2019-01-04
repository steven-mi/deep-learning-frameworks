import tensorflow as tf

class ReLU():
    ''' Implements activation function rectified linear unit (ReLU) 
    '''

    def __init__(self):
        self.params = []

    def forward(self, X):
        return tf.nn.relu(X)
    
class Softmax():
    ''' Implements activation function softmax
    '''

    def __init__(self):
        self.params = []

    def forward(self, X):
        return tf.nn.softmax(X)

class Sigmoid():
    ''' Implements activation function sigmoid 
    '''

    def __init__(self):
        self.params = []

    def forward(self, X):
        return tf.nn.sigmoid(X)

class Tanh():
    ''' Implements activation function tanh
    '''

    def __init__(self):
        self.params = []

    def forward(self, X):
        return tf.nn.tanh(X)
    
class Leaky_ReLU():
    ''' Implements activation function leaky rectified linear unit (leaky ReLU) 
    '''

    def __init__(self, alpha = 0.2):
        self.params = []
        self.alpha = 0.2

    def forward(self, X):
        return tf.nn.leaky_relu(X, self.alpha)