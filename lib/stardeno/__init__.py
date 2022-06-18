
# -- gan noise model --
from . import gan_helper_fun as gh

# -- path --
from pathlib import Path

def load_noise_sim(device):
    mdir = str(Path(__file__).parents[0]/ "../../weights")
    model = gh.load_generator2(mdir, device)
    return model
