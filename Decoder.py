import numpy as np
from Masked_Multi_Head import Masked_Multi_Attention
from Add_and_Norm import Add_Norm
from CrossMultiHead import Cross_Multi_Attention
from FeedForward import Feed_forward

class DecoderArchitecture:
    def __init__(self,layer=1):
        self.block=layer
        self.MaskedMulti=None
        self.NormTheMulti=Add_Norm()
        self.Cross_head=None
        self.Cross_Norm=Add_Norm()
        self.feedAfterCross=Feed_forward()
        self.feed_Norm=Add_Norm()

    def CreateDecoder(self,EncoderOutput,target):
        if self.MaskedMulti is None:
            self.MaskedMulti=Masked_Multi_Attention(target)
        self.MaskedMulti_res=self.MaskedMulti.Concatenate(target)
        self.NormTheMulti_res=self.NormTheMulti.AddAndNorm(target,self.MaskedMulti_res)
        
        if self.Cross_head is None:
            self.Cross_head=Cross_Multi_Attention(EncoderOutput)
        self.Cross_head_res=self.Cross_head.Concatenate(EncoderOutput,self.NormTheMulti_res)
        self.Cross_Norm_res=self.Cross_Norm.AddAndNorm(self.NormTheMulti_res,self.Cross_head_res)
        self.feedAfterCross_res=self.feedAfterCross.initializeFeed(self.Cross_Norm_res)
        self.feed_Norm_res=self.feed_Norm.AddAndNorm(self.Cross_Norm_res,self.feedAfterCross_res)
        
        return self.feed_Norm_res

    def initializeDecoder(self,EncoderOutput,target):
        self.Decoder_res=self.CreateDecoder(EncoderOutput,target)
        return self.Decoder_res

    def backpropagate(self,d_output):
        self.d_feed_Norm_res=self.feed_Norm.backpropagate(d_output)
        self.d_feedAfterCross_res=self.d_feed_Norm_res
        self.d_Cross_Norm_res=self.feed_Norm.d_X
       
        self.d_feed_input=self.feedAfterCross.backpropagate(self.d_feedAfterCross_res)
        self.d_Cross_head_res=self.Cross_Norm.backpropagate(self.d_Cross_Norm_res)
        self.d_NormTheMulti_res=self.Cross_Norm.d_X
        self.d_EncoderOutput,self.d_NormTheMulti_res=self.Cross_head.backpropagate(self.d_Cross_head_res)
        self.d_MaskedMulti_res=self.NormTheMulti.backpropagate(self.d_NormTheMulti_res)
        
        self.d_target=self.NormTheMulti.d_X
        self.d_target+=self.MaskedMulti.backpropagate(self.d_MaskedMulti_res)
        return self.d_EncoderOutput,self.d_target

    def update(self,learning_rate):
        self.MaskedMulti.update(learning_rate)
        self.NormTheMulti.update(learning_rate)
        self.Cross_head.update(learning_rate)
        self.Cross_Norm.update(learning_rate)
        self.feedAfterCross.update(learning_rate)
        self.feed_Norm.update(learning_rate)