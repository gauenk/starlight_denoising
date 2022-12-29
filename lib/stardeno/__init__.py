
# -- api --
from . import gan_helper_fun as gh
from . import post_process as pp
from . import utils

# -- path --
import torch as th
from functools import partial
from pathlib import Path

def load_noise_sim(device,squares=False):
    if "cuda" in str(device):
        device = str(device)
    mdir = str(Path(__file__).parents[0]/ "../../weights")
    model = gh.load_generator2(mdir, device)
    if squares:
        model.forward = partial(process_with_squares,model.forward)
    return model


def process_with_squares(model,vid,chunk_size=256):
    H,W = vid.shape[-2:]
    nH = (H-1)//chunk_size+1
    nW = (W-1)//chunk_size+1
    out = th.zeros_like(vid)
    for iHp in range(0,H,chunk_size):
        for iWp in range(0,W,chunk_size):
            iH = min(H-chunk_size,iHp)
            iW = min(W-chunk_size,iWp)
            vid_sq = vid[...,iH:iH+chunk_size,iW:iW+chunk_size]
            out[...,iH:iH+chunk_size,iW:iW+chunk_size] = model(vid_sq)
    return out
