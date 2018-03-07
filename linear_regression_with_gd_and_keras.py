import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

## The noisy dataset
npts = 1000
x = np.linspace(0,10,npts)
y = 9*x + 20 + 5*np.random.randn(npts)

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original dataset')
plt.show()

## Build the Keras model

# - Input layer: 1 node (the 'x' value)
# - Output layer: 1 node (the 'y' value)
# - No hidden layers
# - Activation: 'linear' (since we're using linear regression)

np.random.seed(1)		# fix random seed for reproducibility

input_dim = 1
output_dim = 1
model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

# Compile: use gradient descent with a learning rate of 0.01
# and using mean square error as the loss function.
sgd = SGD(lr=0.01)
model.compile(loss='mse', optimizer=sgd)

# Train: 1000 epochs and use al training data available as the 'batch_size'
num_epochs = 1000
batch_size = x.shape[0]
history = model.fit(x, y, epochs=num_epochs, batch_size=batch_size,verbose=2)

# Print computed weight values
layer = model.layers
w, b = layer[0].get_weights()
print('Weights: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

# Plot error vs epoch and resulting linear regression
y_regr = model.predict(x)

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('MSE vs. epochs')

plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.title('Original data and result from linear regression')
plt.show()

# Prediction
x_pred = np.array([11.0])
y_pred = model.predict(x_pred)
print("Prediction: y = {:.1f}".format(y_pred[0][0]), " for x = {}".format(x_pred[0]))
