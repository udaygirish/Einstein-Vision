import torch

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
import sys
sys.path.append("../../Mono_Depth_Estimation/ZoeDepth/zoedepth")

repo = "isl-org/ZoeDepth"
# # Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# # Zoe_K
# model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)

# Zoe_NK
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("../../P3Data/test_video_frames/frame_0001.png").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor
depth = depth_tensor.cpu().numpy()  
# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# save colored output
fpath_colored = "output_colored.png"
Image.fromarray(colored).save(fpath_colored)
