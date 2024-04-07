import torch

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
import sys
sys.path.append("../Mono_Depth_Estimation/ZoeDepth/")
sys.path.append("../Mono_Depth_Estimation/ZoeDepth/zoedepth")
from PIL import Image
from zoedepth.utils.misc import save_raw_16bit, colorize


def load_model(repo="isl-org/ZoeDepth", model_name = "ZoeD_NK"):
    model = torch.hub.load(repo, model_name, pretrained=True)
    return model

# Possible model names - "ZoeD_N", "ZoeD_K", "ZoeD_NK"

def run_inference(model, image_path):
    image = Image.open(image_path).convert("RGB")
    depth_numpy = model.infer_pil(image)
    depth_tensor = model.infer_pil(image, output_type="tensor")
    #depth_pil = model.infer_pil(image, output_type="pil")
    
    depth = depth_tensor.cpu().numpy()
    return depth 

def save_output(depth, fpath):
    save_raw_16bit(depth, fpath)
    colored = colorize(depth)
    Image.fromarray(colored).save(fpath.replace(".png", "_colored.png"))

def main():
    model = load_model()
    image_path= "../../P3Data/test_video_frames/frame_0001.png"
    depth = run_inference(model, image_path)
    save_output(depth, "../../P3Data/test_video_frames/frame_0001_depth.png")
    
if __name__ == '__main__':
    main()
