import numpy as np


X= [[1,0],[0,0],[1,1],[0,1]]
Y = [1,0,0,1]


np.random.seed(45)
W0 =np.random.uniform(-1, 1, (2, 2))
B0 =np.random.uniform(-1, 1, (2, 1))
W1 =np.random.uniform(-1, 1, (2, 1))
B1 =np.random.uniform(-1, 1, (1, 1))

def sigmoid(x):
    return 1 / (1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)
    
alpha = 0.1
def predict(x:list[int]):
    x = np.array(x).T
    Z0 = (W0 @ x).reshape(2,1) + B0
    A0 = sigmoid(Z0)
    Z1 = (A0.T @ W1) + B1
    A1 = sigmoid(Z1)
    return A1
for epochs in range(100000):
    for i , x in enumerate(X):
        x = np.array(x).reshape(2, 1)

        Z0 = (W0 @ x).reshape(2,1) + B0
        A0 = sigmoid(Z0)
        Z1 = (A0.T @ W1) + B1
        A1 = sigmoid(Z1)
        
        
        dZ1 = (A1 - Y[i]) * sigmoid_derivative(A1)
        
        derivative_B1 = dZ1
        derivative_W1 =  A0 @ dZ1 
        
        dZ0 = (W1 @ dZ1) * sigmoid_derivative(A0)
        
        derivative_W0 = dZ0 @ x.T
        derivative_B0 = dZ0
        
        
        W0 -= alpha*derivative_W0
        B0 -= alpha*derivative_B0
        W1 -= alpha*derivative_W1
        B1 -= alpha*derivative_B1
    



print("Predictions after 10,000 epochs:")
for x in X:
    pred = predict(x)[0][0]
    print(f"Input {x} -> Predicted: {round(pred)} (Target: {Y[X.index(x)]})")