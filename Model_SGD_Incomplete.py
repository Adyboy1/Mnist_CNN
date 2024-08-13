# %%
import numpy as np
import pandas as pd



data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape 

data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0  
_, m_train = X_train.shape


# %%

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

# %%

def ReLU(Z):
    return np.maximum(Z, 0)

def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


# %%

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = tanh(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# %%
def ReLU_deriv(Z):
    return Z > 0

def tanh_deriv(Z):
    return 1 - np.tanh(Z)**2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def categorical_cross_entropy_loss(A3, Y):
    one_hot_Y = one_hot(Y)
    loss = -np.sum(one_hot_Y * np.log(A3)) / Y.size
    return loss


# %%
# Backward propagation
def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    #print(A3.shape)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dZ2 = W3.T.dot(dZ3) * tanh_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

# %%
# Update parameters
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

# Get predictions
def get_predictions(A3):
    return np.argmax(A3, axis=0)

# Calculate accuracy
def get_accuracy(predictions, Y_batch):
    # Calculates accuracy based on the true labels (Y_batch) of the current batch
    correct_predictions = np.sum(predictions == Y_batch)
    batch_size = Y_batch.size
    return correct_predictions / batch_size

# Gradient descent
'''def gradient_descent(X, Y, alpha, epochs):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 100 == 0:
            loss = categorical_cross_entropy_loss(A3, Y)
            print(f"Epoch: {i}, Loss: {loss}")
            predictions = get_predictions(A3)
            print(f"Accuracy: {get_accuracy(predictions, Y)}")
    return W1, b1, W2, b2, W3, b3'''


# %%
import random

def sgd(X_train, Y_train, alpha, epochs, batch_size):
    m = X_train.shape[1]
    W1, b1, W2, b2, W3, b3 = init_params()
    for epoch in range(epochs):
        # Shuffle data
        permutation = np.random.permutation(m)
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[permutation]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            # Select mini-batch
            X_batch = X_train_shuffled[:, i:i+batch_size]
            Y_batch = Y_train_shuffled[i:i+batch_size]
            """X_batch = X_train[:, i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]"""
            
            
            # Forward and backward propagation for the mini-batch
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch)
            
            # Update parameters
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        
        
        if epoch % 10 == 0:
            _, _, _, _, _, A3_train = forward_prop(W1, b1, W2, b2, W3, b3, X_train)
            loss = categorical_cross_entropy_loss(A3_train, Y_train)
            print(f'Epoch {epoch}, Loss: {loss}')
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y_batch))
    
    return W1, b1, W2, b2, W3, b3


# %%
#W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, alpha=0.15, epochs=501)
W1, b1, W2, b2, W3, b3 = sgd(X_train, Y_train, alpha=0.01, epochs=501, batch_size=64)

# %%
# Final weights and biases
F1=W1
F2=W2
F3=W3
FB1=b1
FB2=b2
FB3=b3

# %% [markdown]
# ~85% accuracy


