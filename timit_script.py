import gpu_manager
import sys

gpu_manager.setup_one_gpu()

from marthe.timit_main import *

if __name__ == '__main__':
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else -2
    exp_config = None
    # tests
    if mode == -2: exp_config = TimitExpConfig.grid(
        small_dts=True, epo=1, lr0=np.exp(np.linspace(np.log(0.001), np.log(0.1), 20)).tolist())
    if mode == -1: exp_config = TimitExpConfig(small_dts=True, epo=1)  # test
    # baseline
    if mode == 0: exp_config = TimitExpConfig.grid(lr0=np.exp(np.linspace(np.log(0.001), np.log(0.1), 20)).tolist())  # baseline

    # marthe
    if mode == 1: exp_config = TimitExpConfigMarthe(beta=1.e-5)
    if mode == 2: exp_config = TimitExpConfigMarthe(beta=1.e-6)
    if mode == 3: exp_config = TimitExpConfigMarthe(beta=1.e-7)
    if mode == 4: exp_config = TimitExpConfigMarthe(beta=1.e-8)
    timit_exp(exp_config)