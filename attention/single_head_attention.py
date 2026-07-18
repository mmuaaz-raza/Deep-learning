from math import sqrt

import numpy as np 



Embbedding_d = 128;dL = 4 ; dK=dQ=8 ; dV = 32 ; h=4
Input = np.random.randn(dL,Embbedding_d)
Wo = np.random.randn(h*dV,Embbedding_d)
def softmax(x):
        x -= x.max(axis=1,keepdims=True)
        return (np.exp(x) / (np.sum(np.exp(x),axis=1,keepdims=True)))

def layer_norm(x):
      epsilon = 1e-8
      return (x-np.mean(x,axis=1,keepdims=True)) / np.sqrt(np.var(x,axis=1,keepdims=True)+epsilon)
      

attention_block_outputs = []
for i in range(h):
    Wk = np.random.randn(Embbedding_d,dK)
    Wq = np.random.randn(Embbedding_d,dQ)
    Wv = np.random.randn(Embbedding_d,dV)

    K = Input @ Wk # (dL,dK)
    Q = Input @ Wq
    V = Input @ Wv

    self_attention = (Q @ K.T )/ (dK**0.5) 
    mask = np.tril(np.ones((dL,dL)))
    mask[mask==0] = -np.inf 
    mask[mask==1] = 0
    self_attention += mask 

    self_attention = softmax(self_attention) @ V
    attention_block_outputs.append(self_attention)

attention=np.concatenate(attention_block_outputs,axis=1)
attention @= Wo

Y = attention + Input 
gamma = np.random.randn(Embbedding_d)
beta = np.random.randn(Embbedding_d)
normalized = layer_norm(Y)
Final = gamma*normalized + beta


