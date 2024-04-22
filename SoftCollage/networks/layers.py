import random
import torch
import copy
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def attention(q,k,v,d_k):
    """
    @params:
        q.size()=k.size()=[bsz,hidden_dim]
        v.size()=[bsz,feat_dim]
        d_k=int
    @return:
        Size([bsz,feat_dim])
    """
    return torch.matmul(F.softmax(torch.matmul(q,k.transpose(-1,-2))/d_k,dim=-1),v)



class CudaModule(nn.Module):
    """
    Only top-level API network class should inherit this class
    """
    def __init__(self):
        super(CudaModule,self).__init__()
        self.__cuda_id=None
        
    def return_cuda_id(self):
        return self.__cuda_id
    
    def set_cuda_id(self,x):
        self.__cuda_id=x
        
    def sync_cuda_id(self,x):
        """
        x is the source TENSOR and self is the destination
        """
        if torch.is_tensor(x):
            cuda_id=self.return_cuda_id()
            return x.to(torch.device(f'cuda:{cuda_id}')) if cuda_id!=None else x.to(torch.device('cpu'))
#         else:
#             cuda_id=self.return_cuda_id()
#             x.set_cuda_id(cuda_id)
#             return x.to(torch.device(f'cuda:{cuda_id}')) if cuda_id!=None else x.to(torch.device('cpu'))
        
    
    
class _SentenceEmbedding(nn.Module):
    def __init__(self,feat_dim=2048+512,da=256,r=1,**kargs):
        super(_SentenceEmbedding,self).__init__()
        self.r=r
        self_attention=[nn.Linear(feat_dim,da)]
        self_attention+=[nn.Tanh()]
        self_attention+=[nn.Linear(da,r)]
        self_attention+=[nn.Softmax(dim=0)]
        self.self_attention=nn.Sequential(*self_attention)
        if r>1:
            self.linear=nn.Linear(r*feat_dim,feat_dim)
        del self_attention
        
    def forward(self,x):
        """
        @params:
            x.size()=[bsz=2,feat_dim]
        @return:
            Size([bsz=2,feat_dim])
        """
        y=torch.matmul(self.self_attention(x).transpose(-1,-2),x).reshape(1,-1) #Size([1,r*feat_dim])
        if self.r>1:
            y=self.linear(y)
            
        del x
        return y
    
        

class _SelfAttention(nn.Module):
    def __init__(self,feat_dim=2048+512,out_dim=2048+512,hidden_dim=1024):
        super(_SelfAttention,self).__init__()
        self.Wq=nn.Linear(feat_dim,hidden_dim,bias=False)#query
        self.Wk=nn.Linear(feat_dim,hidden_dim,bias=False)#key
        self.Wv=nn.Linear(feat_dim,out_dim,bias=False) #value
        self.hidden_dim=hidden_dim
        
    def forward(self,fs):
        """
        @params:
            fs.size()=[bsz=2,feat_dim]
        @return:
            Size([bsz=2,feat_dim])
        """
        return attention(self.Wq(fs),self.Wk(fs),self.Wv(fs),self.hidden_dim)
    
    
    
class _ArithmeticMean(nn.Module):
    def __init__(self,**kargs):
        super(_ArithmeticMean,self).__init__()
        
    def forward(self,x,y):
        """
        @params:
            x.size()=[bsz,feat_dim]
            y.size()=[bsz,feat_dim]
        @return:
            Size([bsz,feat_dim]), bsz=1
        """
        return (x+y)/2
        

        
        