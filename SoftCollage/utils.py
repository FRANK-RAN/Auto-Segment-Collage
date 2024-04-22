import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import sys
import time
import json
import copy
import math
import random
import pickle
import logging
import itertools
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from heapq import *
import networks
import losses
import utils




"""
===========================================CONSTANT=====================================================
"""



LOGGER_LEVEL=logging.INFO
BASE_DIR=os.path.dirname(os.path.abspath(__file__))


"""
=============================================CLASS======================================================
"""



class Pack(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return False
    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v
    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack
    
    
    
"""
=============================================FUNCTION====================================================
"""

def load_img(name,dirname,package="Image"):
    """
    @params:
        name='xxxx.jpg'
        dirname=str
    @return:
        Image.open() object
    """
    return Image.open(os.path.join(dirname,name)) if package=='Image' else cv2.imread(os.path.join(dirname,name))



def load_data(config):
    """
    @params:
        config=Pack()
    @return:
        {'xxxx.jpg':Tensor.Size([1,3,H,W]),...}
        {'xxxx.jpg':1.33333,...}
    """
    postfix=config.mode[:2].upper()
    names=json.load(open(os.path.join(config.ICSS_DIR,
                                      f'AIC-{postfix}.json'),'r'))[config.imgset_name]
    
    name2tensor=dict()
    name2ar=dict()
    preprocess=transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    for name in names:
        img=load_img(name,os.path.join(config.ICSS_DIR,f"ICSS-{postfix}/ICSS-{postfix}-Image"))
        name2tensor[name]=(preprocess(img)).unsqueeze(0)
        name2ar[name]=img.size[0]/img.size[1]
    return name2tensor,name2ar


def check_gpu(config):
    return config.use_gpu and torch.cuda.is_available()



def count_num(func):
    nums=[]
    def wrapper(*args,**kargs):
        if len(args)+len(kargs)==2 or len(args)+len(kargs)==4:
            raise Exception(f"len(inputs)={len(args)+len(kargs)}, but 3 is expected.")
            
        nonlocal nums
        thread=args[-1] if len(kargs)==0 else kargs['thread']
        while(thread+1>len(nums)):
            nums.append(-1)
        nums[thread]=(1+nums[thread])%torch.cuda.device_count()
        
#         print(args[1].size() if torch.is_tensor(args[1]) else type(args[1]),'on cuda:',nums[thread],'by thread',thread)
        
        return func(*args,**kargs,cuda_id=nums[thread])
    return wrapper


@count_num
def to_gpu(config,obj,thread=0,cuda_id=0):
    """
    @usage:
        - only config, obj, thread are supposed to be given
    @func:
        - automatically assign the NET/TENSOR on GPU of cuda_id in loop
        - set cuda_id of the NET(NET.cuda_id==TENSOR.device.index)
    """
    print("---------------------------CUDA-----------------------------")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if config.use_gpu and torch.cuda.is_available():
        # if not torch.is_tensor(obj):
        #     obj.set_cuda_id(cuda_id)
        # return obj.to(torch.device(f'cuda:{cuda_id}'))
        return nn.DataParallel(obj,device_ids=[0, 1]).cuda()



def to_cpu(config,obj):
    """
    @func:
        - return the NET/TENSOR on CPU
        - set cuda_id of the NET(NET.cuda_id==TENSOR.device.index)
    """
    if not torch.is_tensor(obj):
        obj.set_cuda_id(None)
    return obj.to(torch.device('cpu'))
    
    
    
def sync_cuda_id(src,dst):
    """
    @func:
        move src TENSOR to the device of dst TENSOR
    @params:
        - src,dst=Torch.Tensor()
    """
    if torch.is_tensor(src) and torch.is_tensor(dst):
        dst_cuda_id=dst.device.index
        if dst_cuda_id==None:
            return src.to(torch.device("cpu"))
        else:
            return src.to(torch.device(f"cuda:{dst_cuda_id}"))
    else:
        raise Exception(f"Inputs are supposed to be Torch.Tensor, while type(src)={type(src)} and type(dst){type(dst)}.")
    
    

def init_G(config):
    ext=getattr(networks,config.extractor_name)(pretrained=config.pretrained,
                                                ar_dim=config.ar_dim,h_dim=config.h_dim,w_dim=config.w_dim,
                                                info_out_dim=config.info_dim)
    if config.freeze_embedding:
        ext.freeze_embedding()
    fus=getattr(networks,config.fusion_name)(plus_name=config.fusion_plus_name,
                                             feat_dim=config.semantic_dim+config.info_dim,
                                             hidden_dim=config.hidden_dim,da=config.da,r=config.r)
    hvc=getattr(networks,config.HV_classifier_name)(input_dim=config.semantic_dim+config.info_dim)
    lrc=getattr(networks,config.LR_classifier_name)(input_dim=2*(config.semantic_dim+config.info_dim))
    
    return [ext,fus,hvc,lrc]



def config_optimizer(config,G:list):
    
    params=filter(lambda x:x.requires_grad,itertools.chain(*[g.parameters() for g in G]))
    if config.optim=='Adam':
        optimizer=torch.optim.Adam(params,lr=config.lr,weight_decay=config.weight_decay)
    elif config.optim=='SGD':
        optimizer=torch.optim.SGD(params,lr=config.lr,
                          momentum=config.momentum,weight_decay=config.weight_decay)
    else:
        raise
    return optimizer



def cal_dis(f1,f2,dis_type=False):
    """
    f1,f2=Torch.Size(bsz,feat_dim)
    bsz=1 or 2
    """
    with torch.no_grad():
        f1,f2=f1[-1],f2[-1]
        if dis_type==None or dis_type=='Euclidean' or dis_type==False:
            dis=torch.sum((f1-f2)**2) # the square of Euclidean
        elif dis_type=='Cosine':
            dis=1-torch.sum(f1*f2)/torch.norm(f1)/torch.norm(f2) # 1-cos()
        elif dis_type=='StdEuclidean':
            dis=torch.sum((f1-f2)**2/torch.var(torch.cat([f1.unsqueeze(0),f2.unsqueeze(0)],dim=0),dim=0,unbiased=False)) # the square of std Euclidean
        return dis
    
    
    
def init_heap(name2node,config):
    nodes=list(name2node.values())
    heap=[]
    for x in nodes:
        for y in nodes:
            if x.name!=y.name:
                heappush(heap,(cal_dis(sync_cuda_id(x.feat,y.feat),y.feat,config.NNP).item(),(x.name,y.name)))
    return heap
    
    
    
def update_heap(heap,nod_new,name2node,config):
    for node in name2node.values():
        if node.name!=nod_new.name:
            heappush(heap,(cal_dis(sync_cuda_id(nod_new.feat,node.feat),node.feat,config.NNP).item(),(nod_new.name,node.name)))
    return heap
    
    
    
def build_tree(config,name2tensor,name2ar,G:list,logger=None):
    """
    @func:
        Do forward
    @params:
        config=Pack()
        name2tensor={'xxxx.jpg':Tensor.Size([1,3,H,W]),...}
        name2ar={'xxxx.jpg':1.33333,...}
        G=[ext,fus,hvc,lrc]
    @return:
        root SoftNode, [ext0,ext1,...]
    @NOTE:
        Only supports GPU by now; To support CPU, just change to_gpu as to_dev
    """
    ext,fus,hvc,lrc=G
    name2node=dict()
    num_gpu=torch.cuda.device_count()
    num_tensor=len(name2tensor)
    
    if logger!=None:
        logger.debug("copy model")
        
    exts=[ext]
    if config.mode=='train':
        exts+=[copy.deepcopy(ext) for i in range(torch.cuda.device_count()-1)]
        for idx_ext in range(1,len(exts)):
            while True:
                exts[idx_ext]=to_gpu(config,exts[idx_ext],thread=0)
                if exts[idx_ext].return_cuda_id()!=ext.return_cuda_id():
                    break
                    
    if logger!=None:
        logger.debug("extract feature")
        
#     to_dev=getattr(utils,'to_gpu' if check_gpu(config) else 'to_cpu' )
    for idx,(name,tensor) in enumerate(name2tensor.items()):
        tensor=to_gpu(config,tensor,thread=0)
        ext_=exts[idx//int(math.ceil(num_tensor/num_gpu))] if config.mode=='train' else exts[0]
        feat=ext_(tensor,config.W,config.H) if not config.gpu_balanced else to_gpu(config,ext_(tensor,config.W,config.H),thread=1)
        name2node[name]=networks.SoftNode(name=name,feat=feat,ar=name2ar[name])
        
    if logger!=None:
        logger.debug("start construct")
        dis_tot=0
        fusion_tot=0
        cls_tot=0
        heap_time=time.time()
    
    heap=init_heap(name2node,config)
    name2used=dict()
    nam1,nam2=None,None
    while(len(name2node)>1):
        
        if logger!=None:
            dis_time=time.time()
            dis_tot+=dis_time-heap_time
            
        while True:
            nam1,nam2=heappop(heap)[-1]
            if nam1 not in name2used and nam2 not in name2used:
                name2used[nam1]=name2used[nam2]=True
                break
        nod1,nod2=name2node[nam1],name2node[nam2]
        nam_new=nam1+'+'+nam2
        
        if logger!=None:
            fusion_time=time.time()
            dis_tot+=fusion_time-dis_time
            
        feat=fus(nod1.feat,nod2.feat)
        feat=feat if not config.gpu_balanced else to_gpu(config,feat,thread=0)
        nod_new=networks.SoftNode(name=nam_new,feat=feat,children=[nod1,nod2])
        
        if logger!=None:
            cls_time=time.time()
            fusion_tot+=cls_time-fusion_time
            
        nod_new.prob_H=F.softmax(hvc(nod_new),dim=1)[0][0] if not config.gpu_balanced else to_gpu(config,F.softmax(hvc(nod_new),dim=1)[0][0],thread=1)
        nod_new.prob_L=F.softmax(lrc(nod_new),dim=1)[0][0] if not config.gpu_balanced else to_gpu(config,F.softmax(lrc(nod_new),dim=1)[0][0],thread=2)
        nod_new.set_cuda_id(nod_new.prob_H.device.index)
        
        
        if logger!=None:
            heap_time=time.time()
            dis_tot+=heap_time-cls_time
            
        name2node.pop(nam1)
        name2node.pop(nam2)
        name2node[nam_new]=nod_new
        heap=update_heap(heap,nod_new,name2node,config)
        
    if logger!=None:
        dis_time=time.time()
        cls_tot+=dis_time-cls_time
        logger.debug(f"cal_dis_tot_time={dis_tot}")
        logger.debug(f"fusion_tot_time={fusion_tot}")
        logger.debug(f"cls_tot_time={cls_tot}")
        
    return list(name2node.values())[0],exts



def reduce_grad(nets):
    """
    @params:
        - nets=[net0,net1,...]
    @func:
        add the grads of all the nets to the net0
    """
    if len(nets)<=1:
        return
    for i in range(1,len(nets)):
        for p,p_ in zip(nets[0].parameters(),nets[i].parameters()):
            if p.requires_grad:
                p.grad+=p_.grad.to(p.device)
            
    del nets[1:]
    
        

def FAST(root,W,H):
    """
    FAST algorithm
    """
    root.cal_ar()
    if W>=root.ar*H:
        W=int(root.ar*H)
    else:
        H=W//root.ar
    root.fast(W,H)
    return root,W,H



def CROP(root,W,H):
    """
    Fast Crop algorithm. Time complexity: O(n)
    """
    root.cal_ar()
    if W>=root.ar*H:
        newH=int(W/root.ar+0.5)
        newW=W
        direction='H'
        crop_ratio=(newH-H)/newH
    else:
        newH=H
        newW=int(root.ar*H)
        direction='W'
        crop_ratio=(newW-W)/newW
    root.fast(newW,newH)
    root.crop(crop_ratio,direction)
    return root



def RESIZE(root,W,H):
    return CROP(root,W,H)


def tree2collage(root,W,H,dirname,algo='CROP',show=False,save=False,save_path='collage.jpg'):
    """
    @params:
        - root=networks.HardNode()
        - W,H=int,int(canvas width and height)
        - algo='FAST' or 'CROP'
    @return:
        - Image, signal
    """
    collage=Image.new("RGB",(W,H))
    #FAST algorithm
    if algo=='FAST':
        root,_,_=FAST(root,W,H)
    elif algo=='CROP' or algo=='RESIZE':
        root=CROP(root,W,H)
    elif algo=='NOCROP' or algo=='NOFAST' or algo=='NORESIZE':
        pass
    else:
        raise Exception(f"algo param should be FAST or CROP or RESIZE or NOFAST or NOCROP rather than {algo}")
    
    #Map tree to collage
    NUM_0node=0 #indicates whether there is one node whose height and width is assigned zero
    nodes=root.return_leaves()
    if algo=='FAST' or algo=='NOFAST':
        for node in nodes:
            if node.w==0 or node.h==0:
                NUM_0node+=1
                continue
            collage.paste(load_img(name=node.name,dirname=dirname).resize((node.w,node.h)),(node.x,node.y))
    elif algo=='CROP' or algo=='NOCROP':
        for node in nodes:
            if node.w==0 or node.h==0 or node.w_c==0 or node.h_c==0:
                NUM_0node+=1
                continue
            raw_img=load_img(name=node.name,dirname=dirname).resize((node.w,node.h))
            if node.w_c<node.w and node.h_c==node.h:
                cropped_img=raw_img.crop(((node.w-node.w_c)//2,0,(node.w-node.w_c)//2+node.w_c,node.h))
            elif node.h_c<node.h and node.w_c==node.w:
                cropped_img=raw_img.crop((0,(node.h-node.h_c)//2,node.w,node.h_c+(node.h-node.h_c)//2))
            else:
                cropped_img=raw_img
            collage.paste(cropped_img,(node.x_c,node.y_c))
    elif algo=='RESIZE' or algo=='NORESIZE':
        for node in nodes:
            if node.w==0 or node.h==0 or node.w_c==0 or node.h_c==0:
                NUM_0node+=1
                continue
            collage.paste(load_img(name=node.name,dirname=dirname).resize((node.w_c,node.h_c)),(node.x_c,node.y_c))
        
    if show:
        plt.figure()
        plt.imshow(collage)
    if save:
        collage.save(save_path)
    return collage,NUM_0node
    
        
def parse_config(config_name):
    config=Pack(json.load(open(os.path.join(BASE_DIR, config_name))))
    return config
        
    
    
def init(config):

#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir,exist_ok=True)
    if check_gpu(config):
        config["gpu_ids"]=os.environ["CUDA_VISIBLE_DEVICES"]
    json.dump(config,open(os.path.join(config.output_dir,'config.json'),'w'))
    return prepare_logger(config)
        
    
    
def save_checkpoint(ckpt_path:str,G:list,opti,epoch:int,n_iter:int,metrics:dict):
    assert len(G)==4
    ckpt={'G1':G[0].state_dict(),
          'G2':G[1].state_dict(),
          'G3':G[2].state_dict(),
          'G4':G[3].state_dict(),
          'optim':opti.state_dict(),
          'metrics':metrics,
          'epoch':epoch,
          'n_iter':n_iter}
    pickle.dump(ckpt,open(ckpt_path,'wb'))
    
    
    
def load_checkpoint(ckpt_path):
    return pickle.load(open(ckpt_path,'rb')) #,map_location=lambda storage,loc:storage.to(torch.device('cuda:0')))
    
    
    
def map_location(state_dict,cuda_module):
    cuda_id=cuda_module.return_cuda_id()
    for k,v in state_dict.items():
        state_dict[k]=v.to(torch.device(f"cuda:{cuda_id}")) if cuda_id!=None else v.to(torch.device('cpu'))
    return state_dict
    
    
def prepare_logger(config):
    
    logger_level=logging.DEBUG if config.debug else logging.INFO
    
    logger=logging.getLogger()
    logger.setLevel(logger_level)
    
    formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    console=logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logger_level)
    logger.addHandler(console)
    
    log_path=os.path.join(config.output_dir,f'{os.path.basename(config.output_dir)}.log')
    file_handler=logging.FileHandler(log_path)
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    from PIL.PngImagePlugin import logger as PIL_logger
    PIL_logger.setLevel(logging.WARNING)
    
    return logger

def remove_logger(allremove=True):
    logger = logging.getLogger()
    sz=0 if allremove else 1
    while len(logger.handlers)>sz:
        logger.removeHandler(logger.handlers[-1])

