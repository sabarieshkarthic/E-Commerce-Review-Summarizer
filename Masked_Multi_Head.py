import numpy as np
from Masked_Single_Attention import MaskedSingleAttention

class Masked_Multi_Attention:
    def __init__(self,X):
        self.row,self.col=X.shape
        self.num_head=4
        self.head_dim=self.col//self.num_head
        
        self.heads=[]
        for i in range(self.num_head):
            self.heads.append(MaskedSingleAttention(X,self.head_dim))

    def Concatenate(self,X):
        MaskedHeadOutput=[]
        for i in range(self.num_head):
            op=self.heads[i].intializeMaskedAttention(X)
            MaskedHeadOutput.append(op)
            
        multi_mha=np.concatenate(MaskedHeadOutput,axis=1)
        return multi_mha
    
    def backpropagate(self,d_output):
        d_MaskedHeadOutput=[]
        for i in range(self.num_head):
            d_MaskedHeadOutput.append(d_output[:,i*self.head_dim:(i+1)*self.head_dim])
        
        dX_total=np.zeros((d_output.shape[0],self.col))
        for i in range(self.num_head):
            back_X=self.heads[i].backpropagate(d_MaskedHeadOutput[i])
            dX_total=dX_total+back_X
            
        return dX_total
    
    def update(self,lr):
        for i in range(self.num_head):
            self.heads[i].update(lr)