
# Auto-Segment-Collage
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
