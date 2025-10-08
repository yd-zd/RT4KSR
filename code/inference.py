# code/inference.py
import os, torch, argparse, pathlib
import numpy as np
from utils import image
import model
from test import load_checkpoint, reparameterize

class Config:
    """Config object that matches the structure from parser.base_parser()"""
    def __init__(self, scale=2, checkpoint_id="rt4ksr_x2", use_rep=True):
        self.seed = 1
        self.dataroot = os.path.join(pathlib.Path.home(), "datasets/image_restoration")
        self.benchmark = ["ntire23rtsr"]
        self.checkpoints_root = "code/checkpoints"
        self.checkpoint_id = checkpoint_id
        
        # model definitions
        self.bicubic = False
        self.arch = "rt4ksr_rep"
        self.feature_channels = 24
        self.num_blocks = 4
        self.act_type = "gelu"
        self.is_train = True  # Must be True first to load training weights
        self.rep = use_rep
        self.save_rep_checkpoint = False
        
        # data
        self.scale = scale
        self.rgb_range = 1.0

def upscale_image(lr_path, sr_path, scale, ckpt_id, use_rep=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- build network using original author's approach -----
    config = Config(scale=scale, checkpoint_id=ckpt_id, use_rep=use_rep)

    # Use original author's dynamic model loading
    net = torch.nn.DataParallel(
        model.__dict__[config.arch](config)
    ).to(device)
    net = load_checkpoint(net, device, config.checkpoint_id)
    
    # Apply reparameterization if requested (like original test.py)
    if config.rep:
        net = reparameterize(config, net, device)
    
    net.eval()

    # ----- read LR image -----
    lr_uint = image.imread_uint(lr_path, n_channels=3)           # HWC uint8 RGB
    lr_tensor = image.uint2tensor4(lr_uint).to(device)           # BCHW float32 [0,1]

    # ----- inference -----
    with torch.no_grad():
        sr_tensor = net(lr_tensor)

    # ----- postprocess like test.py -----
    sr_tensor = sr_tensor * 255.0

    # ----- save SR -----
    sr_uint = image.tensor2uint(sr_tensor)                       # back to uint8 RGB
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(sr_path)
    if output_dir:  # Only create directory if there's actually a directory path
        os.makedirs(output_dir, exist_ok=True)
    
    image.imsave(sr_uint, sr_path)
    print(f"Saved: {sr_path}")

def generate_output_path(input_path, scale):
    """Generate output path by appending scale factor to input filename"""
    input_pathlib = pathlib.Path(input_path)
    stem = input_pathlib.stem  # filename without extension
    suffix = input_pathlib.suffix  # file extension
    return str(input_pathlib.parent / f"{stem}--{scale}x{suffix}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='path to low-res image')
    p.add_argument('--output', help='where to save the upscaled result (optional, will be auto-generated if not provided)')
    p.add_argument('--scale', type=int, default=2, choices=[2,3])
    p.add_argument('--no-rep', action='store_true', help='disable reparameterization (slower but matches training)')
    args = p.parse_args()

    # Generate output path if not provided
    if args.output is None:
        args.output = generate_output_path(args.input, args.scale)

    ckpt = f"rt4ksr_x{args.scale}"
    upscale_image(args.input, args.output, args.scale, ckpt, use_rep=not args.no_rep)