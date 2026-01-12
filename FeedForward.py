import numpy as np
import pandas as pd

class Feed_forward:
    def define_weights(self,x,y):
        lmt=np.sqrt(6/(x+y))
        arr=np.random.uniform(-lmt,lmt,size=(x,y))
        return arr
    
    def relu(self,X):
        return np.maximum(0,X)
    def d_relu(self,X):
        return np.where(X>0,1,0)
    def Feed(self,inp,w1,w2,b1,b2):
        self.inp=inp
        self.w1=w1
        self.w2=w2
        self.b1=b1
        self.b2=b2   
        self.z1=np.dot(self.inp,self.w1)+self.b1
        self.r1=self.relu(self.z1)
        self.z2=np.dot(self.r1,self.w2)+self.b2
        return self.z2
    def intializeFeed(self,X):
        self.inp=X
        self.row,self.No_of_neurons=X.shape
        if not hasattr(self,"w1"):
            self.w1=self.define_weights(self.No_of_neurons,4*self.No_of_neurons)
            self.w2=self.define_weights(4*self.No_of_neurons,self.No_of_neurons)
            self.b1=np.zeros((1,4*self.No_of_neurons))
            self.b2=np.zeros((1,self.No_of_neurons))
        res=self.Feed(self.inp,self.w1,self.w2,self.b1,self.b2)
        return res
    def backpropagate(self,d_out):
        self.d_w2=np.dot(self.r1.T,d_out)
        self.d_b2=np.sum(d_out,axis=0,keepdims=True)
        self.d_r1=np.dot(d_out,self.w2.T)
        self.d_z1=self.d_r1*self.d_relu(self.r1)
        self.d_w1=np.dot(self.inp.T,self.d_z1)
        self.d_b1=np.sum(self.d_z1,axis=0,keepdims=True)
        self.d_inp=np.dot(self.d_z1,self.w1.T)
        return self.d_inp
    
    def update(self,lr):
        self.w1=self.w1-lr*self.d_w1
        self.w2=self.w2-lr*self.d_w2
        self.b1=self.b1-lr*self.d_b1
        self.b2=self.b2-lr*self.d_b2
        
        
        
        
        
        
        
        
        
        
        
        