import tensorflow as tf

class NeuralNetwork:
    ''' Creates a neural network from a given layer architecture 

    This class is suited for fully connected network and
    convolutional neural network architectures. It connects 
    the layers and passes the data from one end to another.
    '''

    def __init__(self, layers):
        ''' Setup a global parameter list and initilize a
            score function that is used for predictions.

        Args:
            layer: neural network architecture based on layer and activation function objects
            score_func: function that is used as classifier on the output
        '''
        self.layers = layers
        self.params = []
        for layer in self.layers:
            if len(layer.params) > 0:
                self.params.append(layer.params)

    def forward(self, X):
        ''' Pass input X through all layers in the network 
        '''
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict(self, X):
        ''' Run a forward pass and use the score function to classify 
            the output.
        '''
        temp = tf.placeholder(tf.float32, X.shape)
        pred = self.forward(temp)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            pred = sess.run(pred, feed_dict={temp: X})
        return np.argmax(pred, axis=1)