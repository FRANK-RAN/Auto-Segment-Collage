
# Auto-Segment-Collage
Result slide & video demo： https://docs.google.com/presentation/d/16oAbMgt4gp9p5RhnQj_Zkb__c0S4eJXOKm8AwNTVCLU/edit?usp=sharing
video: https://youtu.be/oJaiR_W0OiU
## How to use our web application
1. clone repo```
https://github.com/FRANK-RAN/Auto-Segment-Collage.git```
2. ```cd Auto-Segment-Collage```
3. ```pip install -r requirements. txt```
   
4.  Download SAM
```bash
# Install SAM Packages
pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision

# Download Model weights
mkdir sam_model    # make your sam model dir
cd sam_model
mkdir -p weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./weights
```
```python
# SAM CheckPoint Path is here 
sam_model/weights/sam_vit_h_4b8939.pth
```
Replace CHECKPOINT_PATH in pipeline.py line 88 with your local path  
5. ```python app.py```
6. Demo dataset locates at:
```
/input/custom_dataset_10
```
