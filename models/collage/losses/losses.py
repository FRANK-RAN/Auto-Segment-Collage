from . import rewards
from .rewards import *

class PGLoss(nn.Module):
    def __init__(self,config):
        super(PGLoss,self).__init__()
        self.config=config
        
    def forward(self,sn,hn):
        """
        @params:
            - sn=networks.SoftNode()
            - hn=networks.HardNode()
        @return:
            - Torch.Size([1])
        """
        reward=0.
        for reward_name in self.config.reward_names:
            reward+=getattr(rewards,reward_name)(hn,self.config)
        
        return -reward*sn.cal_loglikelihood(hn)
        