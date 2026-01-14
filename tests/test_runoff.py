import numpy as np

from swimrs.model import runoff


class DummyDay:
    pass


class Dummy:
    pass


def test_runoff_infiltration_excess_zero_when_dry():
    foo = Dummy()
    foo.ksat_hourly = np.zeros((24, 1))
    d = DummyDay()
    d.hr_precip = np.zeros((24, 1))
    runoff.runoff_infiltration_excess(foo, d)
    assert foo.sro.item() == 0.0


def test_runoff_curve_number_reasonable():
    foo = Dummy()
    foo.cn2 = np.array([[75.0]])
    foo.rew = 10.0
    foo.tew = 30.0
    foo.depl_surface = 5.0
    foo.irr_flag = np.array([[0]])
    d = DummyDay()
    d.precip = np.array([[5.0]])
    runoff.runoff_curve_number(foo, d)
    assert np.isfinite(foo.sro).all()
    assert foo.sro.item() >= 0.0
