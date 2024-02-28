import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from huggingface_hub import hf_hub_download
from briarmbg import BriaRMBG
from PIL import Image
import cv2

# Load the RMBG model
net = BriaRMBG()
model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
else:
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()

# Function to resize the image
def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

# Function to process the image
def process(image):
    # Convert the original image to an instance
    orig_image = Image.fromarray(image)
    w, h = orig_image_size = orig_image.size
    
    # Resize the image
    image = resize_image(orig_image)
    im_np = np.array(image)
    
    # Convert the image to a Tensor
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = torch.divide(im_tensor, 255.0)
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()

    # Prediction
    result = net(im_tensor)
    result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
    pil_im = Image.fromarray(np.squeeze(im_array))
    new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    new_im.paste(orig_image, mask=pil_im)
    return new_im

# Function to read the image
def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The specified image was not found or could not be read.")
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_rgb

# Image path and output path
image_path = "input.jpg"
output_path = "output.png"

# Read and process the image and Save the result
img = read_image(image_path)
process(img).save(output_path)
