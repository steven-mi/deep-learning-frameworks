import tensorflow as tf

def cross_entropy(X, y):
    ''' Computes loss and prepares dout for backprop 

    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    '''
    cross_entropy = -tf.reduce_mean(y * tf.log(X)) * 10
    return cross_entropy