# Auto-Segment-Collage

download animal90 dataset
```bash

```


## Download SAM
```bash
# Install SAM Packages
pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision

# Download Model weights
mkdir sam_model    # make your sam model dir
mkdir -p weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./weights
```

```python
# SAM CheckPoint Path is here 
sam_model/weights/sam_vit_h_4b8939.pth
```


Replace CHECKPOINT_PATH in pipeline.py line 88 with your local path
