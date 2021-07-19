
import torch
from torch import cuda
import torch.onnx
import onnx
import numpy as np

onnxfile = "AdaInStyleTransfer_ver11.onnx"
onnx_model = onnx.load(onnxfile)
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession(onnxfile)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
batch_size = 1
device = torch.device("cuda")
#x = torch.randn(batch_size, 3, 256, 256, requires_grad=False, device=device) # content
#y = torch.randn(batch_size, 3, 256, 256, requires_grad=False, device=device) # style

# transform image data to tensor (need 256*256 image)
from torchvision import transforms
def test_transform(size=256):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

content_tf = test_transform(256)
style_tf = test_transform(256)

content_path = './image/house.jpg'
style_path = './image/starrynight.jpg'

#swap: content_path, style_path = (style_path, content_path)

from PIL import Image
content = content_tf(Image.open(str(content_path)))
style = style_tf(Image.open(str(style_path)))

style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)

for ort_input in ort_session.get_inputs():
    name = ort_input.name
    shape = ort_input.shape
    intype = ort_input.type
    print(name, shape, intype)

ort_outs = ort_session.run(None, {'content': to_numpy(content), 'style': to_numpy(style)})
output = np.array(ort_outs)[0]
print(output.shape)

t = torch.from_numpy(output)
print(type(t), t.size())
from torchvision.utils import save_image
save_image(t, './image/outputfromonnx.png')