"""
Simulate Noisy Samples
"""

# -- linalg --
import torch as th
from einops import repeat,rearrange

# -- data --
import data_hub

# -- mgmt --
from easydict import EasyDict as edict

# -- starlight --
import stardeno



def main():

    # -- config --
    cfg = edict()
    cfg.dname = "set8"
    cfg.dset = "te"
    cfg.sigma = 30.
    cfg.isize = "512_512"
    cfg.vid_name = "motorbike"
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = 0 if cfg.frame_start == 0 else cfg.frame_start+cfg.nframes-1
    cfg.device = "cuda:0"
    cfg.bw = False

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     cfg.frame_start,cfg.frame_end)
    sample = data[cfg.dset][indices[0]]
    vid = sample['clean']/255.

    # -- add 3 channels --
    if vid.shape[-3] == 1:
        vid = repeat(vid,'t 1 h w -> t r h w',r=3)

    # -- create 4th channel --
    empty = th.zeros_like(vid[...,[0],:,:])
    vid = th.cat([vid,empty],-3)
    vid = vid.to(cfg.device)
    T,C,H,W = vid.shape
    # print("vid.shape: ",vid.shape)

    # -- load sim  --
    model = stardeno.load_noise_sim(vid.device).to(cfg.device)
    sim = model(vid)

    # -- save --
    stardeno.utils.save_video(vid[...,:3,:,:],"output","clean")
    stardeno.utils.save_video(sim[...,:3,:,:],"output","noisy")

if __name__ == "__main__":
    main()
