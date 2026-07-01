def sig(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_der(z):
    return (z > 0).astype(float)
    
np.random.seed(0)
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

n = len(X)
W1 = np.random.randn(2, 2) * 0.1
b1 = np.zeros(2)
W2 = np.random.randn(2, 1) * 0.1
b2 = np.zeros(1)
lr = 0.1
eps = 1e-15

for i in range(10000):
    #Forward pass
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    y_hat = sig(Z2)
    
    #loss
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -np.mean(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
    
    #Backward
    dZ2 = y_hat - Y
    dW2 = A1.T @ dZ2 / n
    db2 = np.mean(dZ2, axis = 0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_der(Z1)
    dW1 = X.T @ dZ1 / n
    db1 = np.mean(dZ1, axis=0)
    
    #Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if i % 1000 == 0:
        print(f"Epoch {i}:\n W1 = {W1} b1 = {b1}\n W2 = {W2} b2 = {b2}\n Loss = {loss}\n")
        print("Predictions:")
        print((y_hat >= 0.5).astype(int))
        
        print("Probabilities:")
        print(y_hat)
        
        print("Hidden activations:")
        print(A1)
