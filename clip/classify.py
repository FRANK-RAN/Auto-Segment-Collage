import os
import clip
import torch
from PIL import Image
import os

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# load dataset
animal90 = {
    "classes": [],
    "images": []
}

# parse classes
for root, dirs, files in os.walk("data/animals"):
    for name in dirs:
        animal90["classes"].append(name)
        
# parse images
for root, dirs, files in os.walk("data/animals"):
    for name in files:
        animal90["images"].append(os.path.join(root, name))

# Prepare the inputs
image_path = animal90["images"][0] ## substitute with the path to your image
print(f"Image path: {image_path}")
image = Image.open(image_path)
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in animal90["classes"]]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{animal90['classes'][index]:>16s}: {100 * value.item():.2f}%")