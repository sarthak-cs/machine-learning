import numpy as np

np.random.seed(42)

# Single-variable Linear Regression
X = np.random.rand(100) * 10
N = np.random.randn(100)
# print(X)
# print()
Y = X * 4 + 3 + N
# print(Y)

w = np.random.randn()
b = np.random.randn()
lr = 0.01
l = len(X)
for i in range(1000):
    y = X * w + b
    loss = np.mean((Y-y)**2)
    dw = (-2 / l) * np.sum(X * (Y - y))
    db = (-2 / l) * np.sum(Y - y)
    w -= lr * dw
    b -= lr * db
    if i % 100 == 0:
        print(f"Epoch {i}: w = {w:.4f} b = {b:.4f} Loss = {loss:.4f}")
print(w)
print(b)
print(loss)


# Multi-variable Linear Regression
X = np.random.randn(100, 3)
N = np.random.randn(100)
Y = 2*X[:,0] - 4*X[:,1] + 7*X[:,2] + 5 + N
# print(X)
# print(Y)
w = np.random.randn(3)
b = np.random.randn()
lr = 0.01
l = len(X)
for i in range(1000):
    y = X @ w + b
    loss = np.mean((Y-y)**2)
    dw = (-2 / l) * (X.T @ (Y - y))
    db = (-2 / l) * np.sum(Y - y)
    w -= lr * dw
    b -= lr * db
    if i % 100 == 0:
        print(f"Epoch {i}: w = {w} b = {b:.4f} Loss = {loss:.4f}")
print(w)
print(b)
print(loss)
