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

    return X, y 



def load2d(test = False, clos = None):
    """ Load data in 2D for ConvNet """
    X, y = load(test = test)
    X = X.reshape(-1,1,96,96)
    return X, y


def data_split(X,y, split):
    #Split data for training and testing according to split
    #X_train, y_train, X_val, y_val, X_test, y_test

    indice = int(X.shape[0] * (1 - split))

    X_train = X[:indice,:]
    y_train = y[:indice,:]

    X_val = X[indice:,:] 
    y_val = y[indice:,:]

    return X_train, y_train, X_val, y_val



def ploting_img(X, y):
    print 'Ploting dat image Bro'
    plt.imshow(X.reshape(96,96), cmap='Greys_r') 
    y = y.reshape(15,2)
    
    plt.scatter(y[:,0],y[:,1], marker='+')
    plt.show()



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
            num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify)


    # 30-unit output layer
    network = lasagne.layers.DenseLayer(
            network,
            num_units=30,
            nonlinearity=None)

    return network


def image_flip(X, y):
    """ Image flip for data augmentation """

    #Flipping images 
    bs = X.shape[0]
    indices  = np.random.choice(bs, bs/2, replace=False) #Half of the images fliped
    X[indices] = X[indices,:,:,::-1]

    #Flipping target
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    if y is not None:
        y[indices, ::2] = y[indices, ::2] * -1 #Flip all 

        for a, b in flip_indices:
            y[indices, a], y[indices, b] = (y[indices, b], y[indices, a])

    return X ,y 



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



def prediction(X):
    """predict with trained network"""
    input_var = T.tensor4('inputs')
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    with np.load('model_convnet.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    prediction = lasagne.layers.get_output(network, deterministic=True)
    prediction = (prediction * 48) + 48
    prediction_function = theano.function([input_var],prediction) 


    return prediction_function(X)


def para_updates(l_rate, moment, epoch, num_epochs):
    #update parameter according to the number of epoch
    # learning rate from 0.03 to 0.0001
    #momentum from 0.9 to 0.999
    l_rate_begin = 0.03
    l_rate_end = 0.0001
    moment_begin = 0.9
    moment_end = 0.999

    l_rate = ((l_rate_begin - l_rate_end) / num_epochs) * epoch + l_rate_begin
    moment = ((moment_begin - moment_end) / num_epochs ) * epoch + moment_begin

    return l_rate, moment 



def main(num_epochs=500):
	# Load the dataset
    print("Loading data...")
    X,y = load2d()
    X_train, y_train, X_val, y_val = data_split(X,y, split = 0.2)

	
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


    #Learning rate decay & Momentum increase
    l_rate_begin = np.float32(0.03)
    l_rate_end = np.float32(0.0001)
    moment_begin = np.float32(0.9)
    moment_end = np.float32(0.999)

    learning_rate = theano.shared(l_rate_begin)
    momentum = theano.shared(moment_begin)

    epo = T.scalar('epo', dtype='float32')
    num_epo = T.scalar('num_epo', dtype='float32')
    
    update_learning_rate = theano.function([epo , num_epo], learning_rate, updates=[(learning_rate, ((l_rate_end - l_rate_begin) / num_epo) * epo + l_rate_begin)])
    update_momentum = theano.function([epo , num_epo], momentum, updates=[(momentum, ((moment_end - moment_begin) / num_epo) * epo + moment_begin)])


    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate.get_value(), momentum=momentum.get_value())

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
    t_loss = np.empty(num_epochs)
    v_loss = np.empty(num_epochs)
    axe_x = np.arange(num_epochs)

   # Launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0

        #Data augmentation with random flip
        X_train, y_train = image_flip(X_train,y_train)

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

        t_loss[epoch] = train_err / train_batches
        v_loss[epoch] = val_err / val_batches

        #Update learning rate and momentum
        update_learning_rate(np.float32(epoch), np.float32(num_epochs))
        update_momentum(np.float32(epoch),np.float32(num_epochs))


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(t_loss[epoch]))
        print("  validation loss:\t\t{:.6f}".format(v_loss[epoch]))
        #print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))


    #Save all network parameters after training
    np.savez('model_simple_preceptron.npz', *lasagne.layers.get_all_param_values(network))

    #with np.load('model_simple_preceptron.npz') as f:
     #   param_values = [f['arr_%d' % i] for i in range(len(f.files))]
      #  lasagne.layers.set_all_param_values(network, param_values)

    #test_prediction = lasagne.layers.get_output(network, deterministic=True)

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
    #Training
	main()
    
    # #Prediction
    # print("Loading data...")
    # X,y = load2d(test = True)
    # print("Making prediction")
    # pred_y =  prediction(X)
    # ploting_img(X[1,0,:,:], np.array(1,1))


