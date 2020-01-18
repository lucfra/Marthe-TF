import gpu_manager

gpu_manager.setup_one_gpu()

from marthe.timit_main import *

if __name__ == '__main__':
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    exp_config = None
    if mode == -1: exp_config = TimitExpConfig(small_dts=True, epo=1)  # test
    if mode == 0: exp_config = TimitExpConfig()  # baseline
    if mode == 1: exp_config = TimitExpConfigMarthe(beta=1.e-5)
    if mode == 2: exp_config = TimitExpConfigMarthe(beta=1.e-6)
    if mode == 3: exp_config = TimitExpConfigMarthe(beta=1.e-7)
    if mode == 4: exp_config = TimitExpConfigMarthe(beta=1.e-8)
    timit_exp(exp_config)