from .layers import *
from . import layers



class SentenceEmbedding(CudaModule):
    def __init__(self,plus_name="",feat_dim=2048+512,da=256,r=1,**kargs):
        super(SentenceEmbedding,self).__init__()
        self.r=r
        self_attention=[nn.Linear(feat_dim,da)]
        self_attention+=[nn.Tanh()]
        self_attention+=[nn.Linear(da,r)]
        self_attention+=[nn.Softmax(dim=0)]
        self.self_attention=nn.Sequential(*self_attention)
        if r>1:
            self.linear=nn.Linear(r*feat_dim,feat_dim)
        
        if len(plus_name)>0:
            self.plus_net=getattr(layers,plus_name)(feat_dim=feat_dim,da=da,r=r,**kargs)
        else:
            self.plus_net=None
        
    def forward(self,x,y):
        """
        @params:
            x.size()=[bsz,feat_dim]
            y.size()=[bsz,feat_dim]
        @return:
            Size([bsz,feat_dim]), bsz=1 or 2
        """
        xx=self.sync_cuda_id(x)
        yy=self.sync_cuda_id(y)
        
        
        #original
        x=torch.cat([xx[0].unsqueeze(0),yy[0].unsqueeze(0)],dim=0)
        y=torch.matmul(self.self_attention(x).transpose(-1,-2),x).reshape(1,-1) #Size([1,r*feat_dim])
        if self.r>1:
            y=self.linear(y)
        #plus
        if self.plus_net!=None:
            y_plus=self.plus_net(xx[-1].unsqueeze(0),yy[-1].unsqueeze(0))
            y=torch.cat([y,y_plus],dim=0)
            del y_plus
            
        del x,xx,yy
        return y
    
    
class SelfAttention(CudaModule):
    def __init__(self,plus_name="",feat_dim=2048+512,hidden_dim=1024,da=256,r=1,**kargs):
        super(SelfAttention,self).__init__()
        self.self_attention=layers._SelfAttention(feat_dim=feat_dim,hidden_dim=hidden_dim,out_dim=feat_dim)
        self.sentence_embedding=layers._SentenceEmbedding(feat_dim=feat_dim,da=da,r=r)
        
        if len(plus_name)>0:
            self.plus_net=getattr(layers,plus_name)(feat_dim=feat_dim,hidden_dim=hidden_dim,da=da,r=r,**kargs)
        else:
            self.plus_net=None
            
    def forward(self,x,y):
        """
        @params:
            x.size()=[bsz,feat_dim]
            y.size()=[bsz,feat_dim]
        @return:
            Size([bsz,feat_dim]), bsz=1 or 2
        """
        xx=self.sync_cuda_id(x)
        yy=self.sync_cuda_id(y)
        #original
        x=torch.cat([xx[0].unsqueeze(0),yy[0].unsqueeze(0)],dim=0)
        y=self.sentence_embedding(self.self_attention(x))
        if self.plus_net!=None:
            y_plus=self.plus_net(xx[-1].unsqueeze(0),yy[-1].unsqueeze(0))
            y=torch.cat([y,y_plus],dim=0)
            del y_plus
            
        del xx,yy,x
        return y
    
    
    
class ArithmeticMean(CudaModule):
    def __init__(self,plus_name="",**kargs):
        super(ArithmeticMean,self).__init__()
        if len(plus_name)>0:
            self.plus_net=getattr(layers,plus_name)(**kargs)
        else:
            self.plus_net=None
        
    def forward(self,x,y):
        """
        @params:
            x.size()=[bsz,feat_dim]
            y.size()=[bsz,feat_dim]
        @return:
            Size([bsz,feat_dim]), bsz=1 or 2
        """
        xx=self.sync_cuda_id(x)
        yy=self.sync_cuda_id(y)
        y=((xx[0]+yy[0])/2).unsqueeze(0)
        
        if self.plus_net!=None:
            y_plus=self.plus_net(xx[-1].unsqueeze(0),yy[-1].unsqueeze(0))
            y=torch.cat([y,y_plus],dim=0)
            del y_plus
            
        del xx,yy
        return y
    
    
    
class Scheme0(CudaModule):
    def __init__(self,feat_dim=2048+512,hidden_dim=1024,r=1,**kargs):
        super(Scheme0,self).__init__()
        self.r=r
        sentence_embedding=[nn.Linear(feat_dim,r)]
        sentence_embedding+=[nn.Softmax(dim=0)]
        self.sentence_embedding=nn.Sequential(*sentence_embedding)
        if r>1:
            self.linear=nn.Linear(r*feat_dim,feat_dim)
        self.self_attention=layers._SelfAttention(feat_dim=feat_dim,hidden_dim=hidden_dim,out_dim=feat_dim)
        
    def forward(self,x,y):
        """
        @params:
            x.size()=[bsz,feat_dim]
            y.size()=[bsz,feat_dim]
        @return:
            Size([bsz,feat_dim]), bsz=1 or 2
        """
        xx=self.sync_cuda_id(x)
        yy=self.sync_cuda_id(y)
        
        
        #original
        x=torch.cat([xx[0].unsqueeze(0),yy[0].unsqueeze(0)],dim=0)
        x=self.self_attention(x)
        y=torch.matmul(self.sentence_embedding(x).transpose(-1,-2),x).reshape(1,-1) #Size([1,r*feat_dim])
        if self.r>1:
            y=self.linear(y)
            
        del x,xx,yy
        return y
    
    
    
class Scheme1(CudaModule):
    def __init__(self,feat_dim=2048+512,hidden_dim=1024,**kargs):
        super(Scheme1,self).__init__()
        sentence_embedding=[nn.Softmax(dim=0)]
        self.sentence_embedding=nn.Sequential(*sentence_embedding)
        self.linear=nn.Linear(feat_dim,feat_dim)
        self.self_attention=layers._SelfAttention(feat_dim=feat_dim,hidden_dim=hidden_dim,out_dim=feat_dim)
        
    def forward(self,x,y):
        """
        @params:
            x.size()=[bsz,feat_dim]
            y.size()=[bsz,feat_dim]
        @return:
            Size([bsz,feat_dim]), bsz=1 or 2
        """
        xx=self.sync_cuda_id(x)
        yy=self.sync_cuda_id(y)
        
        
        #original
        x=torch.cat([xx[0].unsqueeze(0),yy[0].unsqueeze(0)],dim=0)
        x=self.self_attention(x)
        y=torch.matmul(self.sentence_embedding(torch.mean(x,dim=1).unsqueeze(-1)).transpose(-1,-2),x).reshape(1,-1) #Size([1,r*feat_dim])
        y=self.linear(y)
            
        del x,xx,yy
        return y
    
    
class Scheme2(CudaModule):
    def __init__(self,feat_dim=2048+512,hidden_dim=1024,**kargs):
        super(Scheme2,self).__init__()
        sentence_embedding=[nn.Softmax(dim=0)]
        self.sentence_embedding=nn.Sequential(*sentence_embedding)
        self.self_attention=layers._SelfAttention(feat_dim=feat_dim,hidden_dim=hidden_dim,out_dim=feat_dim)
        
    def forward(self,x,y):
        """
        @params:
            x.size()=[bsz,feat_dim]
            y.size()=[bsz,feat_dim]
        @return:
            Size([bsz,feat_dim]), bsz=1 or 2
        """
        xx=self.sync_cuda_id(x)
        yy=self.sync_cuda_id(y)
        
        
        #original
        x=torch.cat([xx[0].unsqueeze(0),yy[0].unsqueeze(0)],dim=0)
        x=self.self_attention(x)
        y=torch.matmul(self.sentence_embedding(torch.mean(x,dim=1).unsqueeze(-1)).transpose(-1,-2),x).reshape(1,-1) #Size([1,r*feat_dim])
            
        del x,xx,yy
        return y
    