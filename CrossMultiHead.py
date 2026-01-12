import numpy as np
from Cross_Attention import CrossAttention

class Cross_Multi_Attention:
    def __init__(self,X):
        self.row,self.col=X.shape
        self.num_head=4
        self.head_dim=self.col//self.num_head
        
        self.heads=[]
        for i in range(self.num_head):
            self.heads.append(CrossAttention(X,self.head_dim))

    def Concatenate(self,K,Q):
        self.K=K
        self.Q=Q
        CrossHeadOutput=[]
        for i in range(self.num_head):
            op=self.heads[i].intializeCrossAttention(K,Q)
            CrossHeadOutput.append(op)
        cross_mha=np.concatenate(CrossHeadOutput,axis=1)
        return cross_mha
    
    def backpropagate(self,d_output):
        d_CrossHeadOutput=[]
        for i in range(self.num_head):
            d_CrossHeadOutput.append(d_output[:,i*self.head_dim:(i+1)*self.head_dim])
        total_K=np.zeros_like(self.K)
        total_Q=np.zeros_like(self.Q)
        for i in range(self.num_head):
            back_K,back_Q=self.heads[i].backpropagate(d_CrossHeadOutput[i])
            total_Q=total_Q+back_Q
            total_K=total_K+back_K
            
        return total_K,total_Q
    
    def update(self,lr):
        for i in range(self.num_head):
            self.heads[i].update(lr)