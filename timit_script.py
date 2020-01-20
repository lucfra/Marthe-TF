import gpu_manager
import sys

gpu_manager.setup_one_gpu()

from marthe.timit_main import *

if __name__ == '__main__':
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    exp_config = None
    # tests
    if mode == -2: exp_config = TimitExpConfig.grid(
        small_dts=True, epo=1, lr0=np.exp(np.linspace(np.log(0.001), np.log(0.1), 20)).tolist())

    seeds = list(range(5))
    if mode == -1: exp_config = TimitExpConfig(small_dts=True, epo=2, lr0=0.2, pat=3)  # test
    # baseline
    if mode == 0: exp_config = TimitExpConfig.grid(lr0=np.exp(np.linspace(np.log(0.001), np.log(.1), 20)).tolist(),
                                                   seed=seeds)  # baseline

    # marthe
    if mode == 1: exp_config = TimitExpConfigMarthe.grid(beta=1.e-5, seed=seeds)  # this seems not to work
    if mode == 2: exp_config = TimitExpConfigMarthe.grid(beta=1.e-6, seed=seeds)
    if mode == 3: exp_config = TimitExpConfigMarthe.grid(beta=1.e-7, seed=seeds)
    if mode == 4: exp_config = TimitExpConfigMarthe.grid(beta=1.e-8, seed=seeds)

    # exponential decay
    # test
    if mode == 10: exp_config = TimitExpConfigExpDecay(dr=.98, small_dts=True, epo=2)

    # HD
    if mode == 20: exp_config = TimitExpConfigHD(small_dts=True, epo=2, beta=1.e-5)

    # RTHO
    if mode == 30: exp_config = TimitExpConfigRTHO(small_dts=True, epo=2, beta=1.e-5)

    # --------------------------------------------------
    timit_exp(exp_config)
