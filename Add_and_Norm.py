import numpy as np
class Add_Norm:
    def __init__(self):
        self.gamma=None
        self.beta=None
    def Layer_Norm(self,sub_out):
        self.sub_out=sub_out
        seq_len,d_model=sub_out.shape
        if self.gamma is None or self.beta is None:
            self.gamma=np.ones((1,d_model))
            self.beta=np.zeros((1,d_model))
            
        self.mean=np.mean(sub_out,axis=1,keepdims=True)
        self.var=np.var(sub_out,axis=1,keepdims=True)
        self.std=np.sqrt(self.var+1e-5)
        self.x_hat=(sub_out-self.mean)/self.std
        self.normed=self.gamma*self.x_hat+self.beta
        return self.normed
    def AddAndNorm(self,X,sub_out):
        self.X=X
        ln=self.Layer_Norm(sub_out)
        self.out=X+ln
        return self.out
    def backpropagate(self,d_out):
        self.d_X=d_out
        d_normed=d_out
        self.d_gamma=np.sum(d_normed*self.x_hat,axis=0,keepdims=True)
        self.d_beta=np.sum(d_normed,axis=0,keepdims=True)
        d_xhat=d_normed*self.gamma
        N=self.sub_out.shape[1]
        
        sum_d_xhat=np.sum(d_xhat,axis=1,keepdims=True)
        sum_d_xhat_xhat=np.sum(d_xhat*self.x_hat,axis=1,keepdims=True)
        d_sub=(1.0/(N*self.std))*(N*d_xhat-sum_d_xhat-self.x_hat*sum_d_xhat_xhat)
        return d_sub
    def update(self,lr):
        self.gamma-=lr*self.d_gamma
        self.beta-=lr*self.d_beta