import numpy as np

class LinearSoftmaxOutput:
    def __init__(self,dim_model,v_size):
        lmt=np.sqrt(6.0/(dim_model+v_size))
        self.W=np.random.uniform(-lmt,lmt,size=(dim_model,v_size))
        self.b=np.zeros((1,v_size))
    def softmax(self,X):
        expX=np.exp(X-np.max(X,axis=1,keepdims=True))
        self.softmaxResult=expX/(np.sum(expX,axis=1,keepdims=True))
        return self.softmaxResult
    def Linear(self,X):
        self.inp=X
        self.line=np.dot(X,self.W)+self.b
        softmax_out=self.softmax(self.line)
        return softmax_out
    
    def backpropagate(self,d_out):
        self.final_out=d_out
        self.d_inp=np.dot(self.final_out,self.W.T)
        self.d_W=np.dot(self.inp.T,self.final_out)
        self.d_b=np.sum(self.final_out,axis=0)
        return self.d_inp
        
    def update(self,lr):
        self.W=self.W-lr*self.d_W
        self.b=self.b-lr*self.d_b