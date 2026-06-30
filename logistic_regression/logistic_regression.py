def sig(z):
    return 1 / (1 + np.exp(-z))
np.random.seed(42)
X = np.random.randn(200,2)
Z = 2*X[:,0] - 3*X[:,1] + 1
Y = (Z > 0).astype(int)
w = np.random.randn(2)
b = np.random.randn()
lr = 0.01
eps = 1e-15
n = len(X)
for i in range(10000):
    z = X @ w + b
    y_hat = sig(z)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    err = y_hat - Y
    loss = -np.mean(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
    dw = X.T @ err / n
    db = np.mean(err)
    w -= lr * dw
    b -= lr * db
    if i % 1000 == 0:
        print(f"Epoch {i}: w = {w} b = {b:.4f} Loss = {loss:.4f}")
        pred = (y_hat >= 0.5).astype(int)
        accuracy = np.mean(pred == Y)
        print(f"Accuracy: {accuracy:.4f}")
        
print(w)
print(b)
print(loss)
