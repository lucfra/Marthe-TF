import gpu_manager
import sys

gpu_manager.setup_one_gpu()

from marthe.timit_main import *

if __name__ == '__main__':
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    exp_config = None

    small = mode < 0
    timit = load_timit(only_primary=True, context=5, small=mode < 0)
    # tests
    if mode == -2: exp_config = TimitExpConfig.grid(
        small_dts=True, epo=1, lr0=np.exp(np.linspace(
            np.log(0.001), np.log(0.1), 20)).tolist())

    seeds = list(range(5))
    if mode == -1: exp_config = TimitExpConfig(small_dts=True, epo=2, lr0=0.2, pat=3)  # test
    # baseline
    if mode == 0: exp_config = TimitExpConfig.grid(lr0=np.exp(np.linspace(np.log(0.001), np.log(.1), 20)).tolist(),
                                                   seed=seeds)  # baseline

    # if mode == 0: exp_config = TimitExpConfig.grid(lr0=np.exp(np.linspace(np.log(0.001), np.log(.1), 20)).tolist(),
    #                                                seed=0)  # baseline

    # marthe
    if mode == 1: exp_config = TimitExpConfigMarthe.grid(beta=1.e-5, seed=seeds)  # this seems not to work
    if mode == 2: exp_config = TimitExpConfigMarthe.grid(beta=1.e-6, seed=seeds)
    if mode == 3: exp_config = TimitExpConfigMarthe.grid(beta=1.e-7, seed=seeds)
    if mode == 4: exp_config = TimitExpConfigMarthe.grid(beta=1.e-8, seed=seeds)

    # exponential decay
    if mode == 10: exp_config = TimitExpConfigExpDecay(dr=.98, small_dts=True, epo=2)
    if mode == 11: exp_config = TimitExpConfigExpDecay.random(
        '12:00:00', 1,  # accounts for 5 runs with 3hr budget
        lr0=lambda rnd: np.exp(np.log(rnd.uniform(0.001, 0.1))),
        dr=lambda rnd: rnd.uniform(0.5, 1.),
        seed=lambda rnd: rnd.choice(seeds),
    )

    # HD
    if mode == 20: exp_config = TimitExpConfigHD(small_dts=True, epo=2, beta=1.e-5)
    if mode == 21: exp_config = TimitExpConfigHD.grid(beta=np.exp(np.linspace(np.log(1.e-8), np.log(1.e-3), 10)).tolist(),
                                                      seed=seeds)
    # RTHO
    if mode == 30: exp_config = TimitExpConfigRTHO(small_dts=True, epo=2, beta=1.e-5)
    if mode == 31: exp_config = TimitExpConfigRTHO.grid(beta=[1.e-5, 1.e-6, 1.e-7, 1.e-8], seed=seeds)

    #
    if mode == 40: exp_config = TimitExpConfigMartheFixedBeta(small_dts=True, epo=2, beta=1.e-5)
    if mode == 41: exp_config = TimitExpConfigMartheFixedBeta.grid(beta=[1.e-4, 1.e-5, 1.e-6, 1.e-7], seed=seeds)
    if mode == 42: exp_config = TimitExpConfigMartheFixedBeta.grid(beta=1.e-4, seed=[1])
    if mode == 43: exp_config = TimitExpConfigMartheFixedBeta.grid(beta=1.e-5, seed=seeds)
    if mode == 44: exp_config = TimitExpConfigMartheFixedBeta.grid(beta=1.e-6, seed=seeds)
    if mode == 45: exp_config = TimitExpConfigMartheFixedBeta.grid(beta=1.e-7, seed=seeds)

    # marthe 2!

    if mode == 99: exp_config = TimitExpConfigMarthe(beta=1.e-5)
    if mode == 100: exp_config = TimitExpConfigMarthe(beta=1.e-6)
    if mode == 101: exp_config = TimitExpConfigMarthe(beta=1.e-7)
    if mode == 102: exp_config = TimitExpConfigMarthe(beta=1.e-8)
    # test mu
    values = [1.e-3 / 5 ** i for i in range(15)]
    if mode == -100: exp_config = TimitExpConfigMarthe.grid(beta=values, epo=2, pat=3, small_dts=True)  # find good beta
    if mode == -101: exp_config = TimitExpConfigMarthe(beta=1.e-6, _verb=True)

    # --------------------------------------------------
    timit_exp(timit, exp_config)
