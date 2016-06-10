""" Facial Keypoints detection form kaggle """

import os 
import time

import numpy as np 
import theano
import theano.tensor as T

import lasagne 
from lasagne import layers
from lasagne.updates import nesterov_momentum

from pandas.io.parsers import read_csv
from sklearn.utils import shuffle 

FTRAIN = '~/Documents/Machine_learning/facialkeypoints/datasets/training.csv'
FTEST = '~/Documents/Machine_learning/facialkeypoints/datasets/test.csv'


def load(test = False, cols = None):
	""" Loading data for either training or testing """

	fname = FTEST if test else FTRAIN 
	df = read_csv(os.path.expanduser(fname))

	#Image column has pixel values separated by space, convert to numpy array
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	if cols:
		df = df[list(cols) + ['Image']]

	#print (df.count()) #print nb of value for each column
	df = df.dropna() #drop all rows with missing values

	X = np.vstack(df['Image'].values) / 255.0  # Scale pixel values to [0,1]
	X = X.astype(np.float32) #convert to correct type 

	if not test:
		y = df[df.columns[:-1]].values #select keypoints as outputs
		y = (y - 48) / 48 #scale coordinates to [-1,1]
		X, y = shuffle(X, y , random_state=42)
		y = y.astype(np.float32)
	else:
		y = None

	#Split data for training and testing(20%) 
	#X_train, y_train, X_val, y_val, X_test, y_test
	X_train = X[:1720,:]
	y_train = y[:1720,:]

	X_val = X[1720:,:] 
	y_val = y[1720:,:]

	return X_train, y_train, X_val, y_val



def load2d(test = False, clos = None):
    """ Load data in 2D for ConvNet """
    X_train, y_train, X_val, y_val = load(test = test)
    X_train = X_train.reshape(-1,1,96,96)
    X_val = X_val.reshape(-1,1,96,96)
    return X_train, y_train, X_val, y_val

def plot_sample(x, y, axis):
    img = x.reshape(96,96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)    



def build_simplelp(input_var=None):
    # This creates an simple perceptron of one hidden layer

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 9216), input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=200, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Finally, we'll add the fully-connected output layer of 30 units:
    l_out = lasagne.layers.DenseLayer(l_hid1_drop, num_units=30, nonlinearity=None)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_cnn(input_var=None):
    # CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer:
    network = lasagne.layers.InputLayer(shape=(None, 1, 96, 96),
                                        input_var=input_var)


    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    # Fully-connected layer
    network = lasagne.layers.DenseLayer(
            network,
            num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 30-unit output layer
    network = lasagne.layers.DenseLayer(
            network,
            num_units=30,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network




################# Batch iterator ###################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(num_epochs=500):
	# Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load2d()

	#print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
	#print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))

    # Prepare Theano variables for inputs and targets
    #input_var = T.fmatrix('inputs')
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    #val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    val_fn = theano.function([input_var, target_var], test_loss)

    #Initialize array to strore trainning and validation loss
    t_loss = np.array(0.0)
    v_loss = np.array(0.0)

   # Launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 200, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 200, shuffle=False):
            inputs, targets = batch
            #err, acc = val_fn(inputs, targets)
            err = val_fn(inputs, targets)
            val_err += err
            #val_acc += acc
            val_batches += 1

        train_loss = train_err / train_batches
        valid_loss = val_err / val_batches
        np.append(t_loss, train_loss)
        np.append(v_loss, valid_loss)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_loss))
        print("  validation loss:\t\t{:.6f}".format(valid_loss))
        #print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    print t_loss
    print v_loss

    #Save all network parameters after training
    np.savez('model_simple_preceptron.npz', *lasagne.layers.get_all_param_values(network))

    # # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))



if __name__ == '__main__':
	main(5)
	



