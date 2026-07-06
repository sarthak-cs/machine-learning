import numpy as np

def softmax(z):
    c = np.max(z, axis = 1, keepdims = True)
    ex = np.exp(z - c)
    tot = np.sum(ex, axis = 1, keepdims = True)
    return ex / tot

def cross(y_hat, Y):
    eps = 1e-9
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -np.sum(Y * np.log(y_hat), axis = 1)
    return np.mean(loss)

def relu(z):
    return np.maximum(0, z)

def relu_der(z):
    return (z > 0).astype(float)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    y_hat = softmax(Z2)

    return Z1, A1, y_hat
    
# np.random.seed(42)

#--------DATA---------

class0 = np.random.randn(100,3) * 3 + [-4,-4, 4]
class1 = np.random.randn(100,3) * 3 + [4,4, -4]
class2 = np.random.randn(100,3) * 3 + [-4,4, -4]
X = np.vstack([class0, class1, class2])

y0 = np.zeros(100, dtype = int)
y1 = np.ones(100, dtype = int)
y2 = np.full(100, 2)
y = np.hstack([y0, y1, y2])

idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

split = int(0.8 * len(X))

X_train = X[:split]
X_test  = X[split:]

y_train = y[:split]
y_test  = y[split:]

Y_train = np.eye(3)[y_train]
Y_test  = np.eye(3)[y_test]

#-------TRAINING--------

hidden = 5 #Hidden neurons
n = len(X_train)
W1 = np.random.randn(3, hidden) * 0.1
b1 = np.zeros((1, hidden))
W2 = np.random.randn(hidden, 3) * 0.1
b2 = np.zeros((1, 3))
lr = 0.1

for i in range(1000):

    Z1, A1, y_hat = forward(X_train, W1, b1, W2, b2)

    loss = cross(y_hat, Y_train)

    #Backward
    dZ2 = y_hat - Y_train
    dW2 = A1.T @ dZ2 / n
    db2 = np.mean(dZ2, axis=0)
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_der(Z1)
    
    dW1 = X_train.T @ dZ1 / n
    db1 = np.mean(dZ1, axis=0)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if i % 250 == 0:
        print(f"Epoch {i}:\n W1 = {W1} b1 = {b1}\n W2 = {W2} b2 = {b2}\n Loss = {loss}\n")

        
print(f"After training:\n W1 = {W1} b1 = {b1}\n W2 = {W2} b2 = {b2}\n Loss = {loss}\n")
        
print("Training Predictions:")
train_pred = np.argmax(y_hat, axis=1)
print(train_pred)
accuracy = np.mean(train_pred == y_train)
print(f"Training Accuracy: {accuracy * 100}%")


#---------- TESTING ----------

Z1, A1, y_hat = forward(X_test, W1, b1, W2, b2)

print("\n\nTesting Predictions:")
test_pred = np.argmax(y_hat, axis=1)
print(test_pred)

loss = cross(y_hat, Y_test)
print(f"\n Testing Loss = {loss}\n")

accuracy = np.mean(test_pred == y_test)

print(f"Test Accuracy: {accuracy * 100}%")
