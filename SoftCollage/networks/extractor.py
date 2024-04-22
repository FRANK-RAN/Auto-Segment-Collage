from .layers import *

class SqueezeNet(CudaModule):
    
    def __init__(self,pretrained,ar_dim=512,w_dim=256,h_dim=256,info_out_dim=512,bsz=1,*args,**kargs):
        """
        feat size=512
        """
        super(SqueezeNet,self).__init__()
        
        self.AR_DIM=ar_dim
        self.W_DIM=w_dim
        self.H_DIM=h_dim
        self.ar_embedding=nn.Parameter(torch.ones(bsz,self.AR_DIM))#nn.Embedding(1,self.AR_DIM,_weight=torch.ones(1,self.AR_DIM))
        self.w_embedding=nn.Parameter(torch.ones(bsz,self.W_DIM))#nn.Embedding(1,self.W_DIM,_weight=torch.ones(1,self.W_DIM))
        self.h_embedding=nn.Parameter(torch.ones(bsz,self.H_DIM))#nn.Embedding(1,self.H_DIM,_weight=torch.ones(1,self.H_DIM))
        
        self.squeeze=nn.Sequential(*list(
            torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=pretrained).children()
                                       )[:-1])
        mlp=[nn.Linear(self.AR_DIM+self.W_DIM+self.H_DIM,info_out_dim)]
        mlp+=[nn.ReLU()]
        self.mlp=nn.Sequential(*mlp)
        
        del mlp
        
    def forward(self,x,W,H):
        """
        @params:
            x.size()=[1,3,h,w]
            
        @return:
            Size([1,feat_dim])
        """
        x=self.sync_cuda_id(x)
        ar=x.size(-1)/x.size(-2)
        feat_img=self.squeeze(x).mean(dim=-1).mean(dim=-1)
        feat_ar=ar*self.ar_embedding
        feat_w=W*self.w_embedding
        feat_h=H*self.h_embedding
        feat_info=torch.cat([feat_ar,feat_w,feat_h],dim=-1)
        feat_info=self.mlp(feat_info)
        feat=torch.cat([feat_img,feat_info],dim=-1)
        
        del x,feat_ar,feat_w,feat_h,feat_img,feat_info
        
        return feat
    
    def freeze_embedding(self):
        self.ar_embedding.requires_grad=False
        self.h_embedding.requires_grad=False
        self.w_embedding.requires_grad=False

class MobileNetv2(CudaModule):
    
    def __init__(self,pretrained,ar_dim=512,w_dim=256,h_dim=256,info_out_dim=512,bsz=1,*args,**kargs):
        """
        feat size=1280
        """
        super(MobileNetv2,self).__init__()
        
        self.AR_DIM=ar_dim
        self.W_DIM=w_dim
        self.H_DIM=h_dim
        self.ar_embedding=nn.Parameter(torch.ones(bsz,self.AR_DIM))#nn.Embedding(1,self.AR_DIM,_weight=torch.ones(1,self.AR_DIM))
        self.w_embedding=nn.Parameter(torch.ones(bsz,self.W_DIM))#nn.Embedding(1,self.W_DIM,_weight=torch.ones(1,self.W_DIM))
        self.h_embedding=nn.Parameter(torch.ones(bsz,self.H_DIM))#nn.Embedding(1,self.H_DIM,_weight=torch.ones(1,self.H_DIM))
        
        self.mobile=nn.Sequential(*list(
            torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained).children()
                                       )[:-1])
        mlp=[nn.Linear(self.AR_DIM+self.W_DIM+self.H_DIM,info_out_dim)]
        mlp+=[nn.ReLU()]
        self.mlp=nn.Sequential(*mlp)
        
        del mlp
        
    def forward(self,x,W,H):
        """
        @params:
            x.size()=[1,3,h,w]
            
        @return:
            Size([1,feat_dim])
        """
        x=self.sync_cuda_id(x)
        ar=x.size(-1)/x.size(-2)
        feat_img=self.mobile(x).mean(dim=-1).mean(dim=-1)
        feat_ar=ar*self.ar_embedding
        feat_w=W*self.w_embedding
        feat_h=H*self.h_embedding
        feat_info=torch.cat([feat_ar,feat_w,feat_h],dim=-1)
        feat_info=self.mlp(feat_info)
        feat=torch.cat([feat_img,feat_info],dim=-1)
        
        del x,feat_ar,feat_w,feat_h,feat_img,feat_info
        
        return feat
    
    def freeze_embedding(self):
        self.ar_embedding.requires_grad=False
        self.h_embedding.requires_grad=False
        self.w_embedding.requires_grad=False

class DenseNet121(CudaModule):
    
    def __init__(self,pretrained,ar_dim=512,w_dim=256,h_dim=256,info_out_dim=512,bsz=1,*args,**kargs):
        """
        feat size=1024
        """
        super(DenseNet121,self).__init__()
        
        self.AR_DIM=ar_dim
        self.W_DIM=w_dim
        self.H_DIM=h_dim
        self.ar_embedding=nn.Parameter(torch.ones(bsz,self.AR_DIM))#nn.Embedding(1,self.AR_DIM,_weight=torch.ones(1,self.AR_DIM))
        self.w_embedding=nn.Parameter(torch.ones(bsz,self.W_DIM))#nn.Embedding(1,self.W_DIM,_weight=torch.ones(1,self.W_DIM))
        self.h_embedding=nn.Parameter(torch.ones(bsz,self.H_DIM))#nn.Embedding(1,self.H_DIM,_weight=torch.ones(1,self.H_DIM))
        
        self.dense=nn.Sequential(*list(
            torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=pretrained).children()
                                       )[:-1])
        mlp=[nn.Linear(self.AR_DIM+self.W_DIM+self.H_DIM,info_out_dim)]
        mlp+=[nn.ReLU()]
        self.mlp=nn.Sequential(*mlp)
        
        del mlp
        
    def forward(self,x,W,H):
        """
        @params:
            x.size()=[1,3,h,w]
            
        @return:
            Size([1,feat_dim])
        """
        x=self.sync_cuda_id(x)
        ar=x.size(-1)/x.size(-2)
        feat_img=self.dense(x).mean(dim=-1).mean(dim=-1)
        feat_ar=ar*self.ar_embedding
        feat_w=W*self.w_embedding
        feat_h=H*self.h_embedding
        feat_info=torch.cat([feat_ar,feat_w,feat_h],dim=-1)
        feat_info=self.mlp(feat_info)
        feat=torch.cat([feat_img,feat_info],dim=-1)
        
        del x,feat_ar,feat_w,feat_h,feat_img,feat_info
        
        return feat
    
    def freeze_embedding(self):
        self.ar_embedding.requires_grad=False
        self.h_embedding.requires_grad=False
        self.w_embedding.requires_grad=False

class ResNet50(CudaModule):
    
    def __init__(self,pretrained,ar_dim=512,w_dim=256,h_dim=256,info_out_dim=512,bsz=1,*args,**kargs):
        """
        feat size=2048
        """
        super(ResNet50,self).__init__()
        
        self.AR_DIM=ar_dim
        self.W_DIM=w_dim
        self.H_DIM=h_dim
        self.ar_embedding=nn.Parameter(torch.ones(bsz,self.AR_DIM))#nn.Embedding(1,self.AR_DIM,_weight=torch.ones(1,self.AR_DIM))
        self.w_embedding=nn.Parameter(torch.ones(bsz,self.W_DIM))#nn.Embedding(1,self.W_DIM,_weight=torch.ones(1,self.W_DIM))
        self.h_embedding=nn.Parameter(torch.ones(bsz,self.H_DIM))#nn.Embedding(1,self.H_DIM,_weight=torch.ones(1,self.H_DIM))
        
        self.resnet=nn.Sequential(*list(
            torchvision.models.resnet50(pretrained=pretrained).children()
                                       )[:-1])
        mlp=[nn.Linear(self.AR_DIM+self.W_DIM+self.H_DIM,info_out_dim)]
        mlp+=[nn.ReLU()]
        self.mlp=nn.Sequential(*mlp)
        
        del mlp
        
    def forward(self,x,W,H):
        """
        @params:
            x.size()=[1,3,h,w]
            
        @return:
            Size([1,feat_dim])
        """
        x=self.sync_cuda_id(x)
        ar=x.size(-1)/x.size(-2)
        feat_img=self.resnet(x).flatten(1)
        feat_ar=ar*self.ar_embedding
        feat_w=W*self.w_embedding
        feat_h=H*self.h_embedding
        feat_info=torch.cat([feat_ar,feat_w,feat_h],dim=-1)
        feat_info=self.mlp(feat_info)
        feat=torch.cat([feat_img,feat_info],dim=-1)
        
        del x,feat_ar,feat_w,feat_h,feat_img,feat_info
        
        return feat
    
    def freeze_embedding(self):
        self.ar_embedding.requires_grad=False
        self.h_embedding.requires_grad=False
        self.w_embedding.requires_grad=False
        
        

class InfoEmbedding(CudaModule):
    
    def __init__(self,ar_dim=512,w_dim=256,h_dim=256,info_out_dim=512,bsz=1,*args,**kargs):
        super(InfoEmbedding,self).__init__()
        
        self.AR_DIM=ar_dim
        self.W_DIM=w_dim
        self.H_DIM=h_dim
        self.ar_embedding=nn.Parameter(torch.ones(bsz,self.AR_DIM))
        self.w_embedding=nn.Parameter(torch.ones(bsz,self.W_DIM))
        self.h_embedding=nn.Parameter(torch.ones(bsz,self.H_DIM))
        
        mlp=[nn.Linear(self.AR_DIM+self.W_DIM+self.H_DIM,info_out_dim)]
        mlp+=[nn.ReLU()]
        self.mlp=nn.Sequential(*mlp)
        
        del mlp
        
    def forward(self,x,W,H):
        """
        @params:
            x.size()=[1,3,h,w]
            
        @return:
            Size([1,feat_dim])
        """
        x=self.sync_cuda_id(x)
        ar=x.size(-1)/x.size(-2)
        feat_ar=ar*self.ar_embedding
        feat_w=W*self.w_embedding
        feat_h=H*self.h_embedding
        feat_info=torch.cat([feat_ar,feat_w,feat_h],dim=-1)
        feat_info=self.mlp(feat_info)
        
        del x,feat_ar,feat_w,feat_h
        
        return feat_info
    
    def freeze_embedding(self):
        self.ar_embedding.requires_grad=False
        self.h_embedding.requires_grad=False
        self.w_embedding.requires_grad=False
                                       
                                       

class PureResNet50(CudaModule):
    
    def __init__(self,pretrained,*args,**kargs):
        super(PureResNet50,self).__init__()
        
        self.resnet=nn.Sequential(*list(
            torchvision.models.resnet50(pretrained=pretrained).children()
                                       )[:-1])
        
    def forward(self,x,W,H):
        """
        @params:
            x.size()=[1,3,h,w]
            
        @return:
            Size([1,feat_dim])
        """
        x=self.sync_cuda_id(x)
        feat_img=self.resnet(x).flatten(1)
        
        return feat_img
    
    def freeze_embedding(self):
        pass
        
        
        