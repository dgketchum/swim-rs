import numpy as np

from swimrs.model import k_dynamics as kd


class Dummy:
    pass


def test_ke_damper_basic():
    swb = Dummy()
    swb.tew = 10.0
    swb.depl_ze = 2.0
    swb.rew = 4.0
    swb.kr_alpha = 0.5
    swb.kc_max = 1.2
    swb.kc_bas = 0.6
    swb.few = 0.3
    swb.ke_max = 0.5

    kd.ke_damper(swb)
    assert np.isfinite(swb.kr).all()
    assert 0.0 <= swb.ke <= swb.ke_max


def test_ks_damper_basic():
    swb = Dummy()
    swb.taw = 150.0
    swb.raw = 60.0
    swb.depl_root = 80.0
    swb.ks_alpha = 0.2

    kd.ks_damper(swb)
    assert np.isfinite(swb.ks).all()
    assert 0.0 <= swb.ks <= 1.0
