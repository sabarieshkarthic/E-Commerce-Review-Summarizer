import numpy as np
from Multi_Head_Attention import Multi_Attention
from FeedForward import Feed_forward
from Add_and_Norm import Add_Norm

class EncoderArchitecture:
    def __init__(self,layer=1):
        self.block=layer
        self.MhaObj=None
        self.Mha_norm=Add_Norm()
        self.FeedObj=None
        self.Feed_norm=Add_Norm()

    def createEncoder(self,X):
        if self.MhaObj is None:
            self.MhaObj=Multi_Attention(X)
        concat_res=self.MhaObj.Concatenate(X)
        self.Mha_norm_res=self.Mha_norm.AddAndNorm(X,concat_res)
        
        if self.FeedObj is None:
            self.FeedObj=Feed_forward()
        self.Feed_res=self.FeedObj.initializeFeed(self.Mha_norm_res)
        self.Feed_norm_res=self.Feed_norm.AddAndNorm(self.Mha_norm_res,self.Feed_res)
        return self.Feed_norm_res

    def InitializeEncoder(self,X):
        self.Encoder_res=self.createEncoder(X)
        return self.Encoder_res

    def backpropagate(self,d_output):
        d_FF_norm=self.Feed_norm.backpropagate(d_output)
        d_FF=self.FeedObj.backpropagate(d_FF_norm)
        d_MHA_norm=self.Mha_norm.backpropagate(d_FF)
        d_X=self.MhaObj.backpropagate(d_MHA_norm)
        self.d_X=d_X
        return d_X

    def update(self,learning_rate):
        self.MhaObj.update(learning_rate)
        self.Mha_norm.update(learning_rate)
        self.FeedObj.update(learning_rate)
        self.Feed_norm.update(learning_rate)