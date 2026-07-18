import numpy as np 



Embbedding_d = 128;dL = 4 ; dK=dQ=24 ; dV = 12
Input = np.random.randn(dL,Embbedding_d)
Wk = np.random.randn(Embbedding_d,dK)
Wq = np.random.randn(Embbedding_d,dQ)
Wv = np.random.randn(Embbedding_d,dV)

K= Input @ Wk # (dL,dK)
Q = Input @ Wq
V = Input @ Wv

self_attention = (Q @ K.T )/ (dK**0.5) 
mask = np.tril(np.ones((dL,dL)))
mask[mask==0] = -np.inf 
mask[mask==1] = 0
self_attention += mask 

def softmax(x):
    x -= x.max(axis=1,keepdims=True)
    return (np.exp(x) / (np.sum(np.exp(x),axis=1,keepdims=True)))




self_attention = softmax(self_attention) @ V
print(self_attention)



