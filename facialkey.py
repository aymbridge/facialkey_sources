""" Facial Keypoints detection form kaggle """


import os 

import numpy as np 
import theano
import theano.tensor as T 
import lasagne 

from pandas.io.parsers import read_csv
from sklearn.utils import shuffle 

FTRAIN = '~/Documents/Machine_learning/facialkeypoints/training.csv'
FTEST = '~/Documents/Machine_learning/facialkeypoints/test.csv'


def load(test = False, cols = None):
	""" Loading data for either training or testing """

	fname = FTEST if test else FTRAIN 
	df = read_csv(os.path.expanduser(fname))

	#Image column has pixel values separated by space, convert to numpy array
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	if cols:
		df = df[list(cols) + ['Image']]

	print (df.count()) #print nb of value for each column
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

if __name__ == '__main__':
	X, y = load()
	print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
	print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))
	