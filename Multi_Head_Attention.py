import numpy as np
from Single_Head_Attention import SingleAttention

class Multi_Attention:
    def __init__(self,X):
        self.row,self.col=X.shape
        self.num_head=4
        self.head_dim=self.col//self.num_head
        
        self.heads=[]
        for i in range(self.num_head):
            self.heads.append(SingleAttention(X,self.head_dim))

    def Concatenate(self,X):
        SingelHeadOutput=[]
        for i in range(self.num_head):
            op=self.heads[i].intializeAttention(X)
            SingelHeadOutput.append(op)
            
        mha=np.concatenate(SingelHeadOutput,axis=1)
        return mha
    
    def backpropagate(self,d_output):
        d_SingleHeadOutput=[]
        for i in range(self.num_head):
            d_SingleHeadOutput.append(d_output[:,i*self.head_dim:(i+1)*self.head_dim])
        
        dX_total=np.zeros((d_output.shape[0],self.col))
        for i in range(self.num_head):
            back_X=self.heads[i].backpropagate(d_SingleHeadOutput[i])
            dX_total=dX_total+back_X
            
        return dX_total
    
    def update(self,lr):
        for i in range(self.num_head):
            self.heads[i].update(lr)