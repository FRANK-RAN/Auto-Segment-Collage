from .layers import *



class SoftNode(CudaModule):
    def __init__(self,name,feat,ar=None,children=[None,None],prob_H=None,prob_L=None):
        """
        @params:
            name=str
            feat=Torch.Size(bsz,feat_dim) #bsz=1 or 2
        @Note:
            If self.feat.size(0)==2, Then self.feat[0]=feat and self.feat[-1]=feat_plus
        
        """
        super(SoftNode,self).__init__()
        self.name=name
        self.feat=feat
        self.ar=ar
        self.children=children
        self.prob_H=prob_H
        self.prob_L=prob_L
        
        if self.prob_H!=None:
            self.set_cuda_id(self.prob_H.device.index)
        
    def predict(self):
        """
        return predicted HardNode
        """
        if self.children[0]==self.children[1]==None:
            return HardNode(name=self.name,ar=self.ar)
        isH=(self.prob_H>0.5)
        isL=(self.prob_L>0.5)
        chs=[self.children[0].predict(),self.children[1].predict()]
        lch,rch=chs if isL else chs[::-1]
        return HardNode(name=self.name,lch=lch,rch=rch,isH=isH,isL=isL,ar=self.ar)
        
    def sample(self):
        """
        return sampled HardNode
        """
        if self.children[0]==self.children[1]==None:
            return HardNode(name=self.name,ar=self.ar)
        test_H,test_L=np.random.uniform(0,1,2)
        isH=True if test_H<self.prob_H else False
        isL=True if test_L<self.prob_L else False
        chs=[self.children[0].sample(),self.children[1].sample()]
        lch,rch=chs if isL else chs[::-1]
        return HardNode(name=self.name,lch=lch,rch=rch,isH=isH,isL=isL,ar=self.ar)
    
    def cal_loglikelihood(self,hn):
        """
        @params:
            - hn=HardNode()
        @return:
            - Torch.Size([])
        @Note:
            p1/p2=exp(log(p1)-log(p2))
        """
        if hn.lch==hn.rch==None:
            return torch.tensor(0.)
        ret=torch.log(self.prob_H) if hn.isH else torch.log(1-self.prob_H)
        ret+=self.sync_cuda_id(torch.log(self.prob_L) if hn.isL else torch.log(1-self.prob_L))
        lch_s,rch_s=self.children if hn.isL else self.children[::-1]
        ret+=self.sync_cuda_id(lch_s.cal_loglikelihood(hn.lch))
        ret+=self.sync_cuda_id(rch_s.cal_loglikelihood(hn.rch))
        return ret
    
    def print_tree(self,tab=0):
        """
        @func:
            visualize the tree with tab key
        """
        if self.children[0]==self.children[1]==None:
            print("\t"*tab+f"SoftNode({self.name})")
            return
        
        print("\t"*tab+f"SoftNode(pH={round(self.prob_H.item(),2)},pL={round(self.prob_L.item(),2)})")
        
        self.children[0].print_tree(tab+1)
        self.children[1].print_tree(tab+1)
        
        
        
class HardNode(object):
    def __init__(self,name,lch=None,rch=None,isH=None,isL=None,ar=None):
        super(HardNode,self).__init__()
        self.name=name
        self.lch=lch
        self.rch=rch
        self.isH=isH
        self.isL=isL
        self.ar=ar
        # for FAST
        self.x=int(0)
        self.y=int(0)
        self.w=None
        self.h=None
        # for CROP
        self.x_c=int(0)
        self.y_c=int(0)
        self.w_c=None
        self.h_c=None
        
        
    def cal_ar(self):
        """
        @return:
            - the ar(int) of the current node
        """
        if self.lch==None and self.rch==None:
            return self.ar
        lar=self.lch.cal_ar()
        rar=self.rch.cal_ar()
        self.ar=lar*rar/(lar+rar) if self.isH else lar+rar
        return self.ar
    
    def fast(self,w,h,dx=0,dy=0):
        """
        Called after function cal_ar()
        No return, just set x,y,w,h
        """
        self.w=int(w)
        self.h=int(h)
        self.x=int(dx)
        self.y=int(dy)
        if self.isH==True:
            uh=min(max(1,int(w/self.lch.ar)),h-1)
            lh=int(h-uh)
            self.lch.fast(w,uh,dx=self.x,dy=self.y)
            self.rch.fast(w,lh,dx=self.x,dy=self.y+uh)
        elif self.isH==False:
            lw=min(max(1,int(h*self.lch.ar)),w-1)
            rw=int(w-lw)
            self.lch.fast(lw,h,dx=self.x,dy=self.y)
            self.rch.fast(rw,h,dx=self.x+lw,dy=self.y)
            
    def crop(self,crop_ratio,direction:str,dpos=0):
        """
        Called after function fast()
        No return, just set x_c,y_c,w_c,h_c
        """
        if direction=='W':
            dw=int(self.w*crop_ratio)
            self.w_c=max(1,self.w-dw)
            self.h_c=self.h
            self.y_c=self.y
            self.x_c=self.x-dpos
            if self.lch!=None and self.rch!=None:
                dw_l=self.lch.crop(crop_ratio,direction,dpos)
                dw_r=self.rch.crop(crop_ratio,direction,dpos if self.isH else dpos+dw_l)
            return dw
        elif direction=='H':
            dh=int(self.h*crop_ratio)
            self.w_c=self.w
            self.h_c=max(1,self.h-dh)
            self.y_c=self.y-dpos
            self.x_c=self.x
            if self.lch!=None and self.rch!=None:
                dh_u=self.lch.crop(crop_ratio,direction,dpos)
                dh_l=self.rch.crop(crop_ratio,direction,dpos+dh_u if self.isH else dpos)
            return dh
        else:
            raise Exception(f"direction param should be 'H' or 'W' rather than {direction}")
            
    def return_leaves(self):
        """
        @return:
            [HardNode(),...]
        """
        if self.lch==self.rch==None:
            return [copy.deepcopy(self)]
        l_leaves=self.lch.return_leaves()
        r_leaves=self.rch.return_leaves()
        return l_leaves+r_leaves
    
    def print_tree(self,show_ar=True,show_xy=True,show_wh=True,show_xy_c=True,show_wh_c=True,tab=0):
        """
        @func:
            visualize the tree with tab key
        """
        if self.lch==self.rch==None:
            print("\t"*tab+self.name,end='')
            if show_ar and self.ar!=None:
                print(f"(ar={round(self.ar,1)})",end='')
            if show_xy and self.w!=None:
                print(f"(x={self.x},y={self.y})",end='')
            if show_wh and self.w!=None:
                print(f"(w={self.w},h={self.h})",end='')
            if show_xy_c and self.w_c!=None:
                print(f"(x_c={self.x_c},y_c={self.y_c})",end='')
            if show_wh_c and self.w_c!=None:
                print(f"(w_c={self.w_c},h_c={self.h_c})",end='')
            print()
            return
        
        print("\t"*tab+'H' if self.isH else "\t"*tab+'V',end='')
        print('L' if self.isL else 'R',end='')
        
        if show_ar and self.ar!=None:
            print(f"(ar={round(self.ar,1)})",end='')
        if show_xy and self.w!=None:
            print(f"(x={self.x},y={self.y})",end='')
        if show_wh and self.w!=None:
            print(f"(w={self.w},h={self.h})",end='')
        if show_xy_c and self.w_c!=None:
            print(f"(x_c={self.x_c},y_c={self.y_c})",end='')
        if show_wh_c and self.w_c!=None:
            print(f"(w_c={self.w_c},h_c={self.h_c})",end='')
        print()
        self.lch.print_tree(show_ar,show_xy,show_wh,show_xy_c,show_wh_c,tab+1)
        self.rch.print_tree(show_ar,show_xy,show_wh,show_xy_c,show_wh_c,tab+1)
        
        