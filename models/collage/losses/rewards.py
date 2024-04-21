from .metrics import *
    
    
    
def size_vanish_reward(hn,config,*args,**kargs):
    return -2 if size_vanish(hn,config,*args,**kargs) else 0
    
    
    
def crop_ratio_reward(hn,config,*args,**kargs):
    """
    @params:
        - hn=networks.HardNode()
        - config=Pack()
    @return:
        - float
    """
    r=crop_ratio(hn,config,*args,**kargs)
    if r>0.5:
        reward=-2
    elif r>0.1:
#         reward=-12.5*(r-0.1)*(r-0.1)
        reward=-5*r+0.5
    elif r>0.001:
        reward=-1-math.log10(r)
    else:
        reward=2
    return reward*config.crop_weight

def crop_ratio_reward_v2(hn,config,*args,**kargs):
    """
    @params:
        - hn=networks.HardNode()
        - config=Pack()
    @return:
        - float
    """
    r=crop_ratio(hn,config,*args,**kargs)
    if r>0.1:
        reward=-2
    elif r>0.001:
        reward=-4-2*math.log10(r)
    else:
        reward=2
    return reward*config.crop_weight


def crop_ratio_reward_v3(hn,config,*args,**kargs):
    """
    @params:
        - hn=networks.HardNode()
        - config=Pack()
    @return:
        - float
    """
    r=crop_ratio(hn,config,*args,**kargs)
    if r>0.5:
        reward=-2
    elif r>0.1:
        reward=-4*r
    elif r>0.001:
        reward=-1.6-1.2*math.log10(r)
    else:
        reward=2
    return reward*config.crop_weight



def aesthetics_awareness_reward(hn,config,*args,**kargs):
    """
    @params:
        - hn=networks.HardNode()
        - config=Pack()
    @return:
        - float
    """
    c_aes=aesthetics_awareness(hn,config,*args,**kargs)
    return c_aes*config.aesthetics_weight