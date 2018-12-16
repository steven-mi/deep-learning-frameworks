import tensorflow as tf
from tqdm import tqdm

class Optimizer():

    def get_minibatches(X, y, batch_size):
        ''' Decomposes data set into small subsets (batch)
        '''
        m = X.shape[0]
        batches = []
        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size, :, :, :]
            y_batch = y[i:i + batch_size, ]
            batches.append((X_batch, y_batch))
        return batches

    def calculate_gradient(network, loss_function):
        grad = []
        for param in network.params:
            W, b = param
            dW = tf.gradients(loss_function, W)
            db = tf.gradients(loss_function, b)
            grad.append([dW, db])
        return grad

    def sgd(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.01, X_test=None, y_test=None, verbose=None):
        ''' Optimize a given network with stochastic gradient descent 
        '''
        X_shape, y_shape = [None], [None]
        for x in X_train.shape[1:]:
            X_shape.append(x)
        for y in y_train.shape[1:]:
            y_shape.append(y)
        X = tf.placeholder(tf.float32, X_shape)
        Y = tf.placeholder(tf.float32, y_shape)

        loss = loss_function(network.forward(X), Y)
        grads = Optimizer.calculate_gradient(network, loss)

        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        for i in range(epoch):
            if verbose:
                print('Epoch', i + 1)
            for X_mini, y_mini in tqdm(minibatches):
                for param, grad in zip(network.params, grads):
                    with tf.Session() as sess:
                        init = tf.global_variables_initializer()
                        sess.run(init)
                        sess_grad = sess.run(grad, feed_dict={X: X_mini, Y: y_mini})
                    for i in range(len(sess_grad)):
                        param[i] = param[i] - learning_rate * sess_grad[i][0]
            if verbose:
                with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    sess_loss = sess.run(loss, feed_dict={X: X_mini, Y: y_mini})
                train_acc = np.mean(np.argmax(y_train, axis=1) == network.predict(X_train))
                test_acc = np.mean(np.argmax(y_test, axis=1) == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(
                    sess_loss, train_acc, test_acc))
        return network