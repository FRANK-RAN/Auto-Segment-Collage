import os
import sys
import copy
import argparse
import logging
import time
import json
import torch
import shutil
import numpy as np

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import networks
import losses
from utils import *

def train_imgset(config):
    """
    @func:
        called only when config.general_level==0, which means model learns on only one imgset given in config
    @NOTE:
        Only supports GPU by now
    """
    logger=init(config)
    
    postfix=config.mode[:2].upper()
    
    start_time = time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))
    logger.info('[START TRAINING]\n{}\n{}'.format(start_time, '=' * 90))
    logger.info(config)
    
    logger.info('loading data...')
    
    name2tensor,name2ar=load_data(config)
    
    for n,t in name2tensor.items():
        logger.info(f"{n}:{t.size()}")
        
    logger.info("Initialzing gpu...")
    if not check_gpu(config):
        raise Exception("GPUs are not ready!")
    logger.info("Initialzing G...")
    G=init_G(config)
    G=[to_gpu(config,net,thread=0) for net in G]
        
    logger.info("Initialzing optimizer...")
    optimizer=config_optimizer(config,G)
    optimizer.zero_grad()

    logger.info("Initialzing loss and metrics...")
    criterion=losses.PGLoss(config)
    metrics=losses.Metrics(config)
    
    start_n_iter=0
    flag=0
    if config.resume==True:
        logger.info(f'load checkpoint from {config.ckpt_path}...')
        ckpt=load_checkpoint(config.ckpt_path)
        start_n_iter=ckpt['n_iter']+1
        for idx,g in enumerate(G):
            g.load_state_dict(ckpt[f'G{idx+1}'])
        optimizer.load_state_dict(ckpt['optim'])
        metrics.assign_metrics(ckpt['metrics'])
    
    logger.info('start iteration...')
    for n_iter in range(start_n_iter,config.total_iteration):
        
        #use G to build softtree
        logger.debug("build_tree[train]")
        softroot,exts=build_tree(config,name2tensor,name2ar,G,logger)
        
        #sample hardtrees from softtree
        logger.debug("sample and resize")
        hns=[RESIZE(softroot.sample(),config.W,config.H) for i in range(config.sample_num)]
        
        #calculate loss
        logger.debug("cal loss")
        L=sum([criterion(softroot,hn) for hn in hns])/config.sample_num

        logger.info(f"{n_iter}-th iter - mean[-Reward(tree_i)log(P(tree_i;theta)),1<=i<={config.sample_num}]={float(L)}")
        
        if torch.isnan(L):
            break
        
        logger.debug("backward")
        L.backward()
        logger.debug("reduce_grad")
        reduce_grad(exts)
        
        logger.debug("optimizer.step")
        optimizer.step()
        optimizer.zero_grad()
        
        if (1+n_iter)%config.check_period==0:
            with torch.no_grad():
                logger.debug("build_tree[test]")
                softroot,_=build_tree(config,name2tensor,name2ar,G)
                logger.debug("predict")
                hardroot=softroot.predict()
                logger.debug("RESIZE")
                hardroot=RESIZE(hardroot,config.W,config.H)
                logger.debug("metrics.is_better")
                if metrics.is_better(hardroot):
                    result=metrics.return_metrics()
                    logger.info("better result is found:")
                    logger.info(result)
                    if config.save_ckpt:
                        save_checkpoint(ckpt_path=os.path.join(config.output_dir,'ckpt.pkl'),
                                        G=G,
                                        opti=optimizer,
                                        epoch=n_iter,
                                        n_iter=n_iter,
                                        metrics=result)
                    if config.save_best_output:
                        logger.debug("tree2collage[NORESIZE]")
                        _,flag=tree2collage(hardroot,config.W,config.H,algo='NORESIZE',dirname=os.path.join(
                            config.ICSS_DIR,f"ICSS-{postfix}/ICSS-{postfix}-Image"),save=True,
                                     save_path=os.path.join(config.output_dir,'best_resize.png'))
    
    metrics.save_metrics(os.path.join(config.output_dir,'metrics_crop.json'))
    logger.info('exit 0.')
    
    remove_logger()
    del name2tensor,name2ar,G,optimizer,criterion
    return metrics,flag
            
            
def main():
    config=parse_config()
    train_imgset(config)

    
def collage(config):
    
    LAST_IMGSET_NAME=config.LAST_IMGSET_NAME
    START=False
    sict=json.load(open(os.path.join(config.ICSS_DIR, f"selected_images.json"),'r'))
        
    img_erosion={}
    for imgset_name,imgs in sict.items():
        
        config["imgset_name"]=imgset_name
        config["output_dir"]=f"{config.root_output_dir}/{imgset_name}"

        # if len(LAST_IMGSET_NAME)!=0 and config.imgset_name==LAST_IMGSET_NAME:
        #     START=True
        #     if os.path.exists(config.output_dir):
        #         shutil.rmtree(config.output_dir)
        #         print(f"rm -rf {config.output_dir}")
        # if len(LAST_IMGSET_NAME)!=0 and not START:
        #     print(f"skip for output_dir {config.output_dir}")
        #     continue

        try:
            m,flag=train_imgset(config)
            if flag>0:
                img_erosion[imgset_name]=flag
        except KeyError:
            print(f"No imgset_name={config.output_dir}")
            if os.path.exists(config.output_dir):
                shutil.rmtree(config.output_dir)
                print(f"rm -rf {config.output_dir}")
            continue
    json.dump(img_erosion,open(os.path.join(config.root_output_dir,'img_erosion.json'),'w'))

if __name__=='__main__':
    # main()
    collage()
    