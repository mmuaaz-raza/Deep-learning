import numpy as np

print(np.array([[2],[4]]) * np.array([[9,8],[2,1]]))

rng = np.random.default_rng()

Input = rng.uniform(-1,1,size=(3,128))
Wq = rng.uniform(-1,1,size=(128,32))
Wk = rng.uniform(-1,1,size=(128,32))
Wv = rng.uniform(-1,1,size=(128,128))

# forward pass

Iwq = Input @ Wq
Iwk = Input @ Wk
Wkq = (Iwk @ Iwq.T )/32 

denominators = np.zeros(10)
for i,col in enumerate(Wkq.T):
    denominators[i] = np.sum(np.exp(col),axis=0)

for i in range(Wkq.shape[0]):
    for j in range(Wkq.shape[1]) :
        if i < j:
            Wkq[i][j]  = -np.inf   
        else :
            Wkq[i][j] = np.exp(Wkq[i][j]) / denominators[j]  

V = Input @ Wv

Va = Wkq * V 



