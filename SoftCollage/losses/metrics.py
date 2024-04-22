import os
import math
import numpy as np
from scipy.spatial import distance
import torch
import json
from PIL import Image
import torch.nn as nn
import cv2



def size_vanish(hn,config,*args,**kargs):
    """
    return bool
    """
    wh_l=np.array([np.array([node.w,node.h]) for node in hn.return_leaves()])
    return bool((wh_l<20).any())
    


def blankspace_ratio(hn,config,*args,**kargs):
    """
    @params:
        - hn=networks.HardNode()
    """
    return max(0.,1-hn.w_c*hn.h_c/config.W/config.H)



def overlap_ratio(hn,config,*args,**kargs):
    return 0.



def crop_ratio(hn,config,*args,**kargs):
    """
    crop_ratio [CROP] = bsr [FAST]
    """
    if hn.w_c==hn.w:
        return (hn.h-hn.h_c)/hn.h
    elif hn.h_c==hn.h:
        return (hn.w-hn.w_c)/hn.w
    else:
        raise Exception(f"hn.h_c,hn.h,hn.w,hn.w_c={hn.h_c},{hn.h},{hn.w},{hn.w_c}")

        
        
def aspect_ratio_preservation(hn,config,*args,**kargs):
    """
    @NOTE:
        arp [CROP] = 1/(1-bsr) [FAST]
        dH/newH [CROP] = bsr [FAST]
        dW/newW [CROP] = bsr [FAST]
        
        smaller and closer to 1, better
    """
    return 1/(1-crop_ratio(hn,config))



def saliency_awareness(hn,config,*args,**kargs):
    """
    bigger and closer to 1, better
    """
#     ret=0.
#     nodes=hn.return_leaves()
#     for node in nodes:
#         mask=Image.open(os.path.join(config.ICSS_DIR,
#                                          os.path.join(f'ICSS-{config.mode[:2].upper()}/ICSS-{config.mode[:2].upper()}-Mask',node.name[:-3]+'png'))).resize((node.w,node.h))
        
#         if node.w_c<node.w and node.h_c==node.h:
#             mask_cropped=mask.crop(((node.w-node.w_c)//2,0,(node.w-node.w_c)//2+node.w_c,node.h))
#             ret+=float(np.sum(np.array(mask_cropped)==255)/np.sum(np.array(mask)==255))
#         elif node.h_c<node.h and node.w_c==node.w:
#             mask_cropped=mask.crop((0,(node.h-node.h_c)//2,node.w,node.h_c+(node.h-node.h_c)//2))
#             ret+=float(np.sum(np.array(mask_cropped)==255)/np.sum(np.array(mask)==255))
#         else:
#             ret+=1.
#     return ret/len(nodes)
    return 0.



def correlation_preservation(hn,config,*args,**kargs):
    """
    smaller and closer to 0, better
    """
    nodes=hn.return_leaves()
    syn2nodes=dict()
    for node in nodes:
        syn=node.name[:9]
        if syn in syn2nodes.keys():
            syn2nodes[syn].append(node)
        else:
            syn2nodes[syn]=[node]
    ret=0.
    for syn,node_l in syn2nodes.items():
        x_syn=sum([node.x_c+node.w_c/2 for node in node_l])/len(node_l)
        y_syn=sum([node.y_c+node.h_c/2 for node in node_l])/len(node_l)
        ret+=sum([math.sqrt(((node.x_c+node.w_c/2-x_syn)/config.W)**2+
                            ((node.y_c+node.h_c/2-y_syn)/config.H)**2) for node in node_l])
    return ret/len(nodes)
        
    

def aesthetics_awareness(hn,config,*args,**kargs):
    """
    closer to 1, better
    """
    def is_adjacent(ni,nj):
        a,b,c,d=ni.x_c,ni.x_c+ni.w_c,ni.y_c,ni.y_c+ni.h_c
        x,y,z,w=nj.x_c,nj.x_c+nj.w_c,nj.y_c,nj.y_c+nj.h_c
        if (a==y or b==x) and (c<z<d or c<w<d):
            return True
        if (d==z or c==w) and (a<x<b or a<y<b):
            return True
        return False
        
    nodes=hn.return_leaves()
    postfix=config.mode[:2].upper()
    imgs=[cv2.cvtColor(load_img(node.name,os.path.join(config.ICSS_DIR,f"ICSS-{postfix}/ICSS-{postfix}-Image"),package='opencv'),cv2.COLOR_BGR2HSV) for node in nodes]
    Hs=[cv2.calcHist(images=[img],channels=[0],mask=None,histSize=[256],ranges=[0,256]) for img in imgs]
    Hs=[cv2.normalize(H,H).flatten() for H in Hs]
    ret=0
    #color balance
    for i in range(len(Hs)):
        for j in range(i+1,len(Hs)):
            if is_adjacent(nodes[i],nodes[j]):
                ret+=distance.euclidean(Hs[i],Hs[j])           
    ret=ret*2/len(Hs)/(len(Hs)-1)
    #DCM
    so1,mo1,so2,mo2=0,0,0,0
    for node in nodes:
        so1+=node.w_c/config.W*(node.x_c+node.w_c//2)/config.W
        so2+=node.h_c/config.H*(node.y_c+node.h_c//2)/config.H
        mo1+=node.w_c/config.W
        mo2+=node.h_c/config.H
    
    
#     print("so1=",so1)
#     print("so2=",so2)
#     print("mo1=",mo1)
#     print("mo2=",mo2)
#     hn.print_tree()
        
        
    ret+=math.sqrt((so1/mo1+so2/mo2))
    return ret



class Metrics(object):
    """
    This class is used to maintain the optimal metric results
    """
    def __init__(self,config):
        
        super(Metrics,self).__init__()
        self.config=config
        self.metrics=None
        self.single_metric=None
        self.vital_metrics=['_'.join(rwdn.split('_')[:-1]) for rwdn in config.reward_names]
        
    def compute_vital_metrics(self,hn):
        ret=dict()
        from . import metrics
        for metric_name in self.vital_metrics:
            res=getattr(metrics,metric_name)(hn,self.config)
            if isinstance(res,dict):
                for res_k,res_v in res.items():
                    ret[res_k]=res_v
            else:
                ret[metric_name]=res
        return ret
        
    def compute_metrics(self,hn,start_dict={}):
        ret=start_dict
        from . import metrics
        for metric_name in self.config.metric_names:
            if metric_name not in ret.keys():
                res=getattr(metrics,metric_name)(hn,self.config)
                if isinstance(res,dict):
                    for res_k,res_v in res.items():
                        ret[res_k]=res_v
                else:
                    ret[metric_name]=res
        return ret
    
    def _merge_metrics(self,metrics_dict):
        ret=0
        for metric_name,metric_value in metrics_dict.items():
            if metric_name=="crop_ratio":
                ret+=(1-metric_value)*self.config.crop_weight
            elif metric_name=='aesthetics_awareness':
                ret+=metric_value*self.config.aesthetics_weight
            elif metric_name=='size_vanish':
                ret+=-1000 if metric_value else 0
        return ret
            
    def _compare_metrics(self,hn,vital_metrics_dict):
        single_metric=self._merge_metrics(vital_metrics_dict)
        if self.metrics==None or single_metric>self.single_metric:
            if 'size_vanish' in vital_metrics_dict.keys():
                if vital_metrics_dict['size_vanish']:
                    return False
            self.metrics=self.compute_metrics(hn,vital_metrics_dict)
            self.single_metric=single_metric
            return True
        else:
            return False
            
            
    def is_better(self,hn):
        """
        @params:
            - hn=networks.HardNode()
        """
        vital_metrics_dict=self.compute_vital_metrics(hn)
        return self._compare_metrics(hn,vital_metrics_dict)
    
    def return_metrics(self):
        return {'metrics':self.metrics,'single_metric':self.single_metric}
    
    def assign_metrics(self,met):
        self.metrics=met['metrics']
        self.single_metric=met['single_metric']
        
    def save_metrics(self,pathname):
        json.dump({'metrics':self.metrics,'single_metric':self.single_metric},open(pathname,'w'))
        
        
def load_img(name,dirname,package="Image"):
    """
    @params:
        name='xxxx.jpg'
        dirname=str
    @return:
        Image.open() object
    """
    return Image.open(os.path.join(dirname,name)) if package=='Image' else cv2.imread(os.path.join(dirname,name))