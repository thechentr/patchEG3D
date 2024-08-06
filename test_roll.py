from datasets.render import EG3DRender
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch  
import numpy as np  
from PIL import Image  
import time
import random
import torchvision.transforms.functional as TF

def tensor_to_image(tensor, file_path):  
    """  
    Converts a tensor to an image and saves it as a PNG file.  

    Args:  
        tensor (torch.Tensor): [3, w, h], [0, 1]
        file_path (str): The file path to save the PNG image.  
    """  
    tensor = tensor.cpu().detach().numpy()  
    tensor = np.transpose(tensor, (1, 2, 0))  
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255  
    tensor = tensor.astype(np.uint8)  
    image = Image.fromarray(tensor)  
    image.save(file_path)  

gpu_id = 0
device = torch.device(f'cuda:{gpu_id}')
    
render = EG3DRender(device=device)



idx = random.randint(0, 1000000)
patch_tensor = torch.rand((3, 256, 256), device=device)
opoints = lambda img: [ [0, 0], 
                [0, img.shape[1]], 
                [img.shape[2], img.shape[1]], 
                [img.shape[1], 0] ]


x = list(range(-90, 90, 10))
vs = torch.tensor([15] * len(x)).cuda()
hs = torch.tensor(x).cuda()
states = torch.stack([hs, vs], dim=1)

img_tensor, rpoints = render.reset([idx]*len(x), states)

for i in range(len(x)):
    patch = TF.perspective(patch_tensor, opoints(patch_tensor), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
    img_tensor[i] = torch.where(patch.mean(0) == -1, img_tensor[i], patch)

for i in range(len(x)):
    tensor_to_image(img_tensor[i], "output_image.png")  
    time.sleep(0.3)