import os
from pyemu.utils import PstFrom


def build_pest_etd(model_dir, pest_dir, **kwargs):
    pest = PstFrom(model_dir, pest_dir, remove_existing=True)

    for k, v in kwargs['pars'].items():
        _file = v.pop('file')
        pest.add_parameters(_file, 'constant', **v)

    pest.add_observations(kwargs['obs']['file'], insfile=kwargs['obs']['insfile'])

    d = pest.obs_dfs[0].copy()
    d['weight'].loc[d['obsval'] >= 4.] = 0.0
    d['weight'].loc[d['obsval'] < 4.] = 0.0
    d['weight'][:365] = 0.0
    d['weight'][2329:2388] = 1.0
    pest.obs_dfs[0] = d

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst(write_py_file=False)


if __name__ == '__main__':
    d = '/home/dgketchum/PycharmProjects/et-demands/examples/tongue/'

    pp_dir = os.path.join(d, 'pest')

    # obs_file = os.path.join(pp_dir, 'eta.csv')
    ins = '2100.ins'
    p_file = 'params.csv'

    dct = {'obs': {'file': 'eta.np',
                   'insfile': ins},

           'pars': {
               'aw': {'file': p_file,
                      'initial_value': 145.0, 'lower_bound': 100.0, 'upper_bound': 1000.0,
                      'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': 0},

               'rew': {'file': p_file,
                       'initial_value': 3.0, 'lower_bound': 2.0, 'upper_bound': 6.0,
                       'pargp': 'rew', 'index_cols': 0, 'use_cols': 1, 'use_rows': 1},

               'tew': {'file': p_file,
                       'initial_value': 18.0, 'lower_bound': 6.0, 'upper_bound': 29.0,
                       'pargp': 'tew', 'index_cols': 0, 'use_cols': 1, 'use_rows': 2},

               'ndvi_alpha': {'file': p_file,
                              'initial_value': 0.9, 'lower_bound': 0.1, 'upper_bound': 1.5,
                              'pargp': 'ndvi_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': 3},

               'ndvi_beta': {'file': p_file,
                             'initial_value': 0.9, 'lower_bound': 0.1, 'upper_bound': 1.5,
                             'pargp': 'ndvi_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': 4},

           }
           }
    build_pest_etd(d, pp_dir, **dct)
# ========================= EOF ====================================================================
