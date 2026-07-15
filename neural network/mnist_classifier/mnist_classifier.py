import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
def load_data():
    data = pd.read_csv("./data/train.csv").to_numpy()

    X = data[:, 1:] / 255.0
    Y = data[:,0]
    return X,Y

def load_test_data():
    data = pd.read_csv("./data/test.csv").to_numpy()

    X = data / 255.0
    return X


def relu(x):
    return np.maximum(0,x)
def derivative_relu(x):
    return x > 0


def sigmoid(x):
    x = np.clip(x,-500,500)
    return 1 / (1+np.exp(-x))

def derivative_sigmoid(x):
    return x * (1-x)


def embbed_output(y:int):
    output = np.zeros((10,1),np.float16)
    output[y][0] = 1.0
    return output




def load_params():
    filep = Path("params.npz")
    if filep.exists():
        data = np.load("params.npz")
        W0,B0 = data["W0"] , data["B0"]
        W1,B1 = data["W1"] , data["B1"]
        W2,B2 = data["W2"] , data["B2"]
        return W0,B0,W1,B1,W2,B2
    else : 
        rng = np.random.default_rng()
        rand_limit_0 = math.sqrt(2.0/(28*28))
        rand_limit_1 = math.sqrt(2.0/(128))
        rand_limit_2 = math.sqrt(2.0/(64))

        #  2 hidden layer
        W0 = rng.uniform(-rand_limit_0,rand_limit_0,(128,28*28))
        B0 = np.zeros((128,1))

        W1 = rng.uniform(-rand_limit_1,rand_limit_1,(64,128))
        B1 =np.zeros((64,1))

        #  1 output layer
        W2 = rng.uniform(-rand_limit_2,rand_limit_2,(10,64))
        B2 =np.zeros((10,1))
        return W0,B0,W1,B1,W2,B2

def interpret_prediction(y):
    pred_list:list[tuple[int,float]] = []
    for i,row in enumerate(y):
        pred_list.append((i,row[0]))
    pred_list.sort(key=lambda x:x[1],reverse=True)
    return pred_list[0][0]




X,Y = load_data()
W0,B0,W1,B1,W2,B2 = load_params()

def forward_pass(x):
    Z0 = W0 @ x.reshape(-1,1) + B0
    A0 = relu(Z0)

    Z1 = W1 @ A0 + B1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + B2
    A2 = sigmoid(Z2)
    return A2



alpha = 0.001

def evaluate(X):
    corrects = 0
    total = X.shape[0]
    for i,x in enumerate(X):
        if interpret_prediction(forward_pass(x)) == Y[i]:
            corrects+= 1
    print(f"corrects : {corrects}/{total}")
    return corrects/total


def generateLabel(X):
    data:list = []
    for i,x in enumerate(X):
        data.append([i+1,interpret_prediction(forward_pass(x))])
    
    output= pd.DataFrame(data,columns=["ImageId","Label"])
    output.to_csv("./data/submission.csv", index=False)
    return output


epoch_size =20
for epoch in range(epoch_size):
    if epoch%5 == 0:
        print(f"epoch {epoch} - Accurary : {evaluate(X)*100}")
    for i,x in enumerate(X):
        # forward pass
        Z0 = W0 @ x.reshape(-1,1) + B0
        A0 = relu(Z0)

        Z1 = W1 @ A0 + B1
        A1 = relu(Z1)

        Z2 = W2 @ A1 + B2
        A2 = sigmoid(Z2)
        predicted_val = A2

        # back pass
        cost_derivative = predicted_val - embbed_output(Y[i])
        dZ2 = derivative_sigmoid(predicted_val)

        dA2 = dB2 = dZ2 * cost_derivative
        dW2 = (dA2 @ A1.T )

        dA1 = dB1 = (W2.T @dA2) * derivative_relu(A1)
        dW1 = (dA1 @ A0.T)
        
        dA0 = dB0= (W1.T @dA1) * derivative_relu(A0)
        dW0 = (dA0 @ x.reshape(-1,1).T)

        # updates
        W0 -= alpha*dW0
        B0 -= alpha*dB0
        W1 -= alpha*dW1
        B1 -= alpha*dB1
        W2 -= alpha*dW2
        B2 -= alpha*dB2
    

np.savez("params.npz", W0=W0, B0=B0, W1=W1, B1=B1, W2=W2, B2=B2, learning_rate=alpha, epoch=epoch_size)
    
testX = load_test_data()
generateLabel(testX)



