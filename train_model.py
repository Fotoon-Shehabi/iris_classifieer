%%writefile train_model.py
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

tf.random.set_seed(1)

# load dataset
iris = load_iris()
X = iris['data']
Y = iris['target']

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
encoded_cat_Y = np_utils.to_categorical(encoded_Y)
 
# define model
def build_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
iris_classifier = build_model()
iris_classifier.fit(X, encoded_cat_Y, epochs=64)
