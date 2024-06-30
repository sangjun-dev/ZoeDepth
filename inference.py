import torch
from zoedepth.utils.misc import colorize
from PIL import Image

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Local file
image = Image.open("input/image1.bmp").convert("RGB")  # load
depth = zoe.infer_pil(image)  # as numpy

# Colorize output
colored = colorize(depth)

# save colored output
fpath_colored = "output/image1_colored.png"
Image.fromarray(colored).save(fpath_colored)