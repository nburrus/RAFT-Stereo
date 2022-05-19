import sys
sys.path.append('core')

import argparse
import glob
import gc
from pathlib import Path

import numpy as np
import torch
import torch.onnx

from tqdm import tqdm
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

def export(args, name, devices):    
    # Needs to be /32
    sizes = [(1280, 736), (640,480), (320,256), (160,128)]

    for DEVICE in devices:
        for w,h in sizes:
            gc.collect()
            parallel_model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
            checkpoint = torch.load(args.restore_ckpt)
            parallel_model.load_state_dict(checkpoint)

            model = parallel_model.module
            model.to(DEVICE)
            model.eval()        
            
            with torch.no_grad():
                sample_input = (torch.zeros(1, 3, h, w).to(DEVICE), torch.zeros(1, 3, h, w).to(DEVICE))
                scripted_module = torch.jit.trace(model, sample_input)
                torch.jit.save (scripted_module, f"raft-stereo-{name}-{DEVICE}-{h}x{w}.scripted.pt")
    
    # Need opset 16, not in Pytorch 1.11, only in nightly builds
    # Avoiding the rabbit hole for now.
    # torch.onnx.export(scripted_module,                 # model being run
    #                   sample_input,              # model input (or a tuple for multiple inputs)
    #                   f"raft-stereo.onnx", # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=16,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['left', 'right'],   # the model's input names
    #                   output_names = ['disparity'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        
    # args = parser.parse_args(args=["--restore_ckpt", "models/raftstereo-middlebury.pth", "--corr_implementation", "alt", "--mixed_precision", "--n_downsample", "2"])
    # export(args, 'middlebury', ["cuda"])

    # For some reason corr alt explodes the memory with cpu on my machine. "reg" remains under 20GB.
    args = parser.parse_args(args=["--restore_ckpt", "models/raftstereo-middlebury.pth", "--corr_implementation", "reg", "--mixed_precision", "--n_downsample", "2"])
    export(args, 'middlebury', ["cpu"])

    # args = parser.parse_args(args=["--restore_ckpt", "models/raftstereo-eth3d.pth"])
    # export(args, 'eth3d', ["cpu", "cuda"])

    # args = parser.parse_args(args=[
    #     "--restore_ckpt", "models/raftstereo-eth3d.pth", 
    #     "--corr_implementation", "alt", # for some reason reg does not work with cuda torchscript execution.
    # ])
    # export(args, 'eth3d', ["cuda"])

    # args = parser.parse_args(args=[
    #     "--restore_ckpt", "models/raftstereo-realtime.pth", 
    #     "--shared_backbone", 
    #     "--n_downsample", "3", 
    #     "--n_gru_layers", "2", 
    #     "--slow_fast_gru", 
    #     "--valid_iters", "7", 
    #     "--corr_implementation", "alt", # for some reason reg does not work with cuda torchscript execution.
    #     "--mixed_precision"])
    # export(args, 'fast', ["cuda"])        

    # args = parser.parse_args(args=[
    #     "--restore_ckpt", "models/raftstereo-realtime.pth", 
    #     "--shared_backbone", 
    #     "--n_downsample", "3", 
    #     "--n_gru_layers", "2", 
    #     "--slow_fast_gru", 
    #     "--valid_iters", "7", 
    #     "--corr_implementation", "reg", 
    #     "--mixed_precision"])
    # export(args, 'fast', ["cpu"])        