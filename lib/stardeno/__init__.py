
# -- gan noise model --
import .gan_helper_fun as gh

# -- path --
from pathlib import Path

def load_noise_sim(device):
    mdir = Path(__file__) / "../../" / "weights"
    model = gh.load_generator2(mdir, device)
    return model
