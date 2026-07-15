import math
from pathlib import Path
import pandas as pd
import numpy as np

def load_data0_5():
    data = pd.read_csv("./data/train.csv").to_numpy()
    X = data[:, 1:] / 255.0
    Y = data[:,0]
    mask = Y <=5
    X =X[mask]
    Y =Y[mask]
    return X,Y

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
    filep = Path("params_lora.npz")
    if filep.exists():
        data = np.load("params_lora.npz")
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
    
def load_params_lora(rank:int):
    filep = Path(f"params_lora{rank}.npz")
    if filep.exists():
        data = np.load(f"params_lora{rank}.npz")
        C1,D1 = data["C1"] , data["D1"]
        C0,D0 = data["C0"] , data["D0"]
        return C0,D0,C1,D1
    else : 
        rng = np.random.default_rng()
        mean = 0.0          
        std_dev1 = 1/math.sqrt(784)       
        std_dev2 = 1/math.sqrt(128)       

        
        C0 =np.zeros((128,rank))
        D0 = rng.normal(loc=mean,scale=std_dev1,size=(rank,784))

        C1 =np.zeros((64,rank))
        D1 =rng.normal(loc=mean,scale=std_dev2,size=(rank,128))
        

        
        return C0,D0,C1,D1
    



def interpret_prediction(y):
    pred_list:list[tuple[int,float]] = []
    for i,row in enumerate(y):
        pred_list.append((i,row[0]))
    pred_list.sort(key=lambda x:x[1],reverse=True)
    return pred_list[0][0]




X,Y = load_data0_5()
W0,B0,W1,B1,W2,B2 = load_params()

def forward_pass(x):
    Z0 = W0 @ x.reshape(-1,1) + B0
    A0 = relu(Z0)

    Z1 = W1 @ A0 + B1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + B2
    A2 = sigmoid(Z2)
    return A2

def forward_pass_lora_1(x,C0,D0,C1,D1,s):
    input = x.reshape(-1,1)
    Z0 = W0 @ input + B0 +(s * (C0 @ (D0@input)))
    A0 = relu(Z0)
    CD1 = C1 @ D1
    Z1 = W1 @ A0 + B1 + (s * (CD1@A0))
    A1 = relu(Z1)
    Z2 = W2 @ A1 + B2 
    A2 = sigmoid(Z2)
    
    return A2



alpha = 0.01

def evaluate(X,Y,predict,*args):
    corrects = 0
    total = X.shape[0]
    for i,x in enumerate(X):
        if interpret_prediction(predict(x,*args)) == Y[i]:
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


def TrainNormal(epoch_size:int,W0,B0,W1,B1,W2,B2,X,Y) :
    for epoch in range(epoch_size):
        if epoch%5 == 0:
            print(f"epoch {epoch} - Accurary : {evaluate(X,Y,forward_pass)*100}")
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
        
    return W0,B0,W1,B1,W2,B2



X1,Y1 = load_data()
# W0,B0,W1,B1,W2,B2= TrainNormal(20,W0,B0,W1,B1,W2,B2,X,Y)
# np.savez("params_lora.npz", W0=W0, B0=B0, W1=W1, B1=B1, W2=W2, B2=B2, learning_rate=alpha, epoch=20)


def TrainLora(epoch_size:int,W0,B0,W1,B1,W2,B2,C0,D0,C1,D1,lora_alpha,rank,X,Y , is_bias,is_wights,is_lora):
    s = lora_alpha/rank
    for epoch in range(epoch_size):
        if epoch%5 == 0 or epoch == (epoch_size-1):
            print(f"epoch {epoch} - Accurary : {evaluate(X,Y,forward_pass_lora_1,C0,D0,C1,D1,s)*100}")

        for i,x in enumerate(X):
            # forward pass
            input = x.reshape(-1,1)
            Z0 = W0 @ input + B0 +(s * (C0 @ (D0@input)))
            A0 = relu(Z0)
            CD1 = C1 @ D1
            Z1 = W1 @ A0 + B1 + (s * (CD1@A0))
            A1 = relu(Z1)

            Z2 = W2 @ A1 + B2 
            A2 = sigmoid(Z2)
            predicted_val = A2

            # back pass
            cost_derivative = predicted_val - embbed_output(Y[i])

            dA2 = dB2=  cost_derivative * derivative_sigmoid(predicted_val)
            if is_wights :
                dW2 = dA2 @ A1.T
            if is_lora :
                dA1 = (W2.T @ dA2 )* derivative_relu(A1)
                dD1 = s*( C1.T @ dA1 @ A0.T  )
                dC1 = s* ( dA1 @ (D1@A0).T )

                dA0 =  ((W1 + s*CD1).T @dA1) * derivative_relu(A0)
                dD0 = s*( C0.T @ dA0 @ input.T  )
                dC0 = s* ( dA0 @ (D0@input).T )
            

            
            if is_lora :            
                C1 -= alpha*dC1
                D1 -= alpha*dD1
                C0 -= alpha*dC0
                D0 -= alpha*dD0
            if is_wights :
                W2 -= alpha*dW2
            if is_bias :
                B2 -= alpha*dB2


    return C0,D0,C1,D1,W2,B2

rank = int(input ("rank : "))
is_bias = bool(input ("is bias : "))
is_lora = bool(input ("is lora : "))
is_weights = bool(input ("is weights : "))
print(is_bias,is_lora,is_weights)
C0,D0,C1,D1 = load_params_lora(rank)
C0,D0,C1,D1,W2,B2 = TrainLora(20,W0,B0,W1,B1,W2,B2,C0,D0,C1,D1,1,rank,X1,Y1,is_bias,is_weights,is_lora)
np.savez(f"params_lora{rank}_wieghts.npz", C1=C1,D1=D1,C0=C0,D0=D0,rank=rank)

# W0,B0,W1,B1,W2,B2 = TrainNormal(20,W0,B0,W1,B1,W2,B2)

    
# testX = load_test_data()
# generateLabel(testX)



