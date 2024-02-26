"""crop_cycle.py
Defines DayData class
Defines crop_cycle_mp, crop_cycle, crop_day_loop_mp, crop_day_loop,
    write_crop_output
Called by mod_crop_et.py

"""

import datetime
import logging
import os
import numpy as np
import pandas as pd

from model.etd import compute_field_et
from model.etd.initialize_tracker import PlotTracker
from model.etd import obs_kcb_daily
from model.etd import calculate_height

OUTPUT_FMT = ['et_act',
              'etref',
              'kc_act',
              'kc_bas',
              'ks',
              'ke',
              'ppt',
              'depl_root',
              'depl_surface',
              'irrigation',
              'dperc',
              'fc',
              'few',
              'zr',
              'aw',
              'aw3',
              'taw',
              'p_rz',
              'p_eft',
              'niwr',
              'runoff',
              'irr_day',
              'season',
              'capture',
              ]


class DayData:

    def __init__(self):
        self.etref_array = np.zeros(30)


def field_day_loop(config, plots, debug_flag=False, params=None):
    size = len(plots.input['order'])
    tracker = PlotTracker(size)
    tracker.load_soils(plots)

    # apply calibration parameter updates here
    if config.calibration:
        # PEST++ hacking
        cal_arr = {k: np.zeros((1, size)) for k in config.calibration_groups}

        for k, f in config.calibration_files.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
            idx = plots.input['order'].index(fid)

            if params:
                value = params[k]
            else:
                v = pd.read_csv(f, index_col=None, header=0)
                value = v.loc[0, '1']

            cal_arr[group][0, idx] = value

        for k, v in cal_arr.items():

            tracker.__setattr__(k, v)

            print('{}: {}'.format(k, ['{:.1f}'.format(p) for p in v.flatten()]))

            if k == 'aw':
                tracker.__setattr__('depl_root', tracker.aw / 2)

            if k == 'rew':
                tracker.__setattr__('depl_surface', tracker.tew / 2)
                tracker.__setattr__('depl_zep', tracker.rew / 2)

    targets = plots.input['order']

    # Initialize crop data frame
    if debug_flag:
        tracker.setup_dataframe(targets)
    else:
        eta = []

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0

    refet = None
    ndvi = None
    capture = None

    if config.refet_type == 'eto' and config.field_type == 'irrigated':
        refet = 'eto_mm'
        ndvi = 'ndvi_irr'
        capture = 'etf_irr_ct'

    elif config.refet_type == 'eto' and config.field_type == 'unirrigated':
        refet = 'eto_mm_uncorr'
        ndvi = 'ndvi_inv_irr'
        capture = 'etf_inv_irr_ct'

    elif config.refet_type == 'etr' and config.field_type == 'irrigated':
        refet = 'etr_mm_uncorr'
        ndvi = 'ndvi_irr'
        capture = 'etf_irr_ct'

    elif config.refet_type == 'etr' and config.field_type == 'unirrigated':
        refet = 'etr_mm_uncorr'
        ndvi = 'ndvi_inv_irr'
        capture = 'etf_inv_irr_ct'

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    for step_dt, vals in plots.input['time_series'].items():

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        foo_day.doy = vals['doy']
        foo_day.dt_string = step_dt
        dt = pd.to_datetime(step_dt)
        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.refet = np.array(vals[refet]).reshape(1, -1)
        foo_day.ndvi = np.array(vals[ndvi]).reshape(1, -1)

        if config.field_type == 'irrigated':
            try:
                irr = plots.input['irr_data']
                irr_day = [int(foo_day.doy) in irr[t][str(foo_day.year)]['irr_doys'] for t in targets]
                foo_day.irr_day = np.array(irr_day).reshape(1, -1)

            except KeyError as e:
                print(e)

        foo_day.precip = np.array(vals['prcp_mm'])
        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt
        foo_day.precip = foo_day.precip.reshape(1, -1)

        foo_day.snow_depth = 0.0

        if foo_day.month == 11 and foo_day.day == 1:
            tracker.setup_dormant()

        if foo_day.year == 2020 and foo_day.month == 5 and foo_day.day == 18:
            a = 1

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                tracker.crop_df[fid][step_dt]['capture'] = int(vals[capture][sample_idx[1]])
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_surface'] = tracker.depl_surface[sample_idx]
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

        else:
            eta.append(tracker.etc_act)

    if debug_flag:
        # pass final dataframe to calling script

        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}
        return tracker.crop_df

    else:
        # if not debug, just return the actual ET results as an ndarray
        return np.array(eta)


def write_crop_output(data, et_cell, crop, foo):
    """Write output files for each cell and crop

    Parameters
    ---------
    crop_count : int
        count of crop being computed
    data :

    et_cell :

    crop :

    foo :

    Returns
    -------
    None

    """

    year_field = 'Year'
    month_field = 'Month'
    day_field = 'Day'
    doy_field = 'DOY'

    # Build PMET type fieldname from input data (Eto or ETr)
    et_type = data.refet['fields']['etref']
    pmet_field = 'PM' + et_type
    precip_field = 'PPT'
    etact_field = 'ETact'
    etpot_field = 'ETpot'
    etbas_field = 'ETbas'
    irrig_field = 'Irrigation'
    season_field = 'Season'
    cutting_field = 'Cutting'
    runoff_field = 'Runoff'
    dperc_field = 'DPerc'
    niwr_field = 'NIWR'
    kc_field = 'Kc'
    kcb_field = 'Kcb'
    gs_start_doy_field = 'Start_DOY'
    gs_end_doy_field = 'End_DOY'
    gs_start_date_field = 'Start_Date'
    gs_end_date_field = 'End_Date'
    gs_length_field = 'GS_Length'
    p_rz_field = 'P_rz'
    p_eft_field = 'P_eft'
    p_rz_fraction_field = 'P_rz_fraction'
    p_eft_fraction_field = 'P_eft_fraction'

    columns_order = ['DOY',
                     'PMetr_mm',
                     'ETact',
                     'ETpot',
                     'ETbas',
                     'Kc',
                     'Kcb',
                     'ndvi_irr',
                     'etf_irr'
                     'PPT',
                     'Irrigation',
                     'Runoff',
                     'DPerc',
                     'NIWR',
                     'Season',
                     'Cutting',
                     'P_rz',
                     'P_eft',
                     'ndvi_inv_irr',
                     'Year']

    # Merge crop and weather data frames to form daily output
    if (data.cet_out['daily_output_flag'] or
        data.cet_out['monthly_output_flag'] or
        data.cet_out['annual_output_flag'] or
        data.gs_output_flag):
        # et_cell.crop_coeffs[1].data does not have DateTimeIndex
        add_ = pd.merge(et_cell.climate_df[['ppt']], et_cell.crop_coeffs[1].data,
                        left_index=True, right_index=True)
        daily_output_df = pd.merge(
            foo.crop_df, add_,
            # foo.crop_df, et_cell.climate_df[['ppt', 't30']],
            left_index=True, right_index=True)

        # Rename output columns
        daily_output_df.index.rename('Date', inplace=True)
        daily_output_df[year_field] = daily_output_df.index.year
        daily_output_df = daily_output_df.rename(columns={
            'doy': doy_field, 'ppt': precip_field, 'etref': pmet_field,
            'et_act': etact_field, 'et_pot': etpot_field,
            'et_bas': etbas_field, 'kc_act': kc_field, 'kc_bas': kcb_field,
            'niwr': niwr_field, 'irrigation': irrig_field,
            'runoff': runoff_field, 'dperc': dperc_field,
            'p_rz': p_rz_field, 'p_eft': p_eft_field,
            'season': season_field, 'cutting': cutting_field})

        # TODO: remove this debugging code
        # daily_output_df.dropna(how='any', inplace=True)
        # daily_output_df = daily_output_df[columns_order]
        # daily_output_df = daily_output_df.loc['2011-04-01': '2011-10-31']

    # Compute monthly and annual stats before modifying daily format below
    if data.cet_out['monthly_output_flag']:
        monthly_resample_func = {
            pmet_field: np.sum, etact_field: np.sum, etpot_field: np.sum,
            etbas_field: np.sum, kc_field: np.mean, kcb_field: np.mean,
            niwr_field: np.sum, precip_field: np.sum, irrig_field: np.sum,
            runoff_field: np.sum, dperc_field: np.sum,
            p_rz_field: np.sum, p_eft_field: np.sum,
            season_field: np.sum, cutting_field: np.sum}
        # dri dm approach produces 'TypeError: ("'dict' object is not callable",
        # a u'occurred at index DOY')
        monthly_output_df = daily_output_df.resample('MS').apply(
            monthly_resample_func)
        # add effective ppt fractions to monthly tables
        monthly_output_df[p_rz_fraction_field] = \
            (monthly_output_df[p_rz_field] / monthly_output_df[precip_field]).fillna(0)
        monthly_output_df[p_eft_fraction_field] = \
            (monthly_output_df[p_eft_field] / monthly_output_df[precip_field]).fillna(0)

    if data.cet_out['annual_output_flag']:
        annual_resample_func = {
            pmet_field: np.sum, etact_field: np.sum, etpot_field: np.sum,
            etbas_field: np.sum, kc_field: np.mean, kcb_field: np.mean,
            niwr_field: np.sum, precip_field: np.sum, irrig_field: np.sum,
            runoff_field: np.sum, dperc_field: np.sum,
            p_rz_field: np.sum, p_eft_field: np.sum,
            season_field: np.sum, cutting_field: np.sum}
        # dri dm approach produces 'TypeError: ("'dict' object is not callable",
        # a u'occurred at index DOY')
        annual_output_df = daily_output_df.resample('AS').apply(
            annual_resample_func)
        # add effective ppt fractions to annual tables
        annual_output_df[p_rz_fraction_field] = \
            (annual_output_df[p_rz_field] / annual_output_df[precip_field]).fillna(0)
        annual_output_df[p_eft_fraction_field] = \
            (annual_output_df[p_eft_field] / annual_output_df[precip_field]).fillna(0)

    # Get growing season start and end DOY for each year
    # Compute growing season length for each year
    if data.gs_output_flag:
        # dri dm approach produces 'TypeError: ("'dict' object is not callable",
        # a u'occurred at index DOY')
        gs_output_df = daily_output_df.resample('AS').apply(
            {year_field: np.mean})
        gs_output_df[gs_start_doy_field] = np.nan
        gs_output_df[gs_end_doy_field] = np.nan
        gs_output_df[gs_start_date_field] = None
        gs_output_df[gs_end_date_field] = None
        gs_output_df[gs_length_field] = np.nan
        for year_i, (year, group) in enumerate(daily_output_df.groupby(
            [year_field])):
            if not np.any(group[season_field].values):
                logging.debug('  Skipping, season flag was never set to 1')
                continue
            else:
                season_diff = np.diff(group[season_field].values)
                try:
                    start_i = np.where(season_diff == 1)[0][0] + 1

                    # gs_output_pd.set_value(
                    #     group.index[0], gs_start_doy_field,
                    #     int(group.ix[start_i, doy_field]))
                    # Replacement for set_value Future Warning
                    # gs_output_df.at[group.index[0], gs_start_doy_field] = int(
                    #     group.ix[start_i, doy_field])
                    # Replacement for .ix deprecation 4/22/2020
                    gs_output_df.at[group.index[0], gs_start_doy_field] = int(
                        group.loc[group.index[start_i], doy_field])
                except:
                    # gs_output_pd.set_value(
                    #     group.index[0], gs_start_doy_field,
                    #     int(min(group[doy_field].values)))
                    # Replacement for set_value Future Warning
                    gs_output_df.at[group.index[0], gs_start_doy_field] = int(
                        min(group[doy_field].values))

                try:
                    end_i = np.where(season_diff == -1)[0][0] + 1
                    # gs_output_pd.set_value(
                    #     group.index[0], gs_end_doy_field,
                    #     int(group.ix[end_i, doy_field]))
                    # Replacement for set_value Future Warning
                    # gs_output_df.at[group.index[0], gs_end_doy_field] = int(
                    #     group.ix[end_i, doy_field])
                    # Replacement for .ix deprecation 4/22/2020
                    gs_output_df.at[group.index[0], gs_end_doy_field] = int(
                        group.loc[group.index[end_i], doy_field])
                except:
                    # gs_output_pd.set_value(
                    #     group.index[0], gs_end_doy_field,
                    #     int(max(group[doy_field].values)))
                    # Replacement for set_value Future Warning
                    gs_output_df.at[group.index[0], gs_end_doy_field] = int(
                        max(group[doy_field].values))
                del season_diff
            # gs_output_pd.set_value(
            #     group.index[0], gs_length_field,
            #     int(sum(group[season_field].values)))
            # Replacement for set_value Future Warning
            gs_output_df.at[group.index[0], gs_length_field] = int(
                sum(group[season_field].values))

    base_columns = []
    open_mode = 'w'
    print_index = True
    print_header = True

    # Write daily cet
    if data.cet_out['daily_output_flag']:
        daily_output_df[year_field] = daily_output_df.index.year
        daily_output_df[month_field] = daily_output_df.index.month
        daily_output_df[day_field] = daily_output_df.index.day

        # format date attributes if values are formatted
        if data.cet_out['daily_float_format'] is not None:
            daily_output_df[year_field] = daily_output_df[year_field].map(
                lambda x: ' %4d' % x)
            daily_output_df[month_field] = daily_output_df[month_field].map(
                lambda x: ' %2d' % x)
            daily_output_df[day_field] = daily_output_df[day_field].map(
                lambda x: ' %2d' % x)
            daily_output_df[doy_field] = daily_output_df[doy_field].map(
                lambda x: ' %3d' % x)

        # This will convert negative "zeros" to positive

        daily_output_df[niwr_field] = np.round(daily_output_df[niwr_field], 6)
        # daily_output_df[niwr_field] = np.round(
        # daily_output_df[niwr_field].values, 6)
        daily_output_df[season_field] = daily_output_df[season_field].map(
            lambda x: ' %1d' % x)
        daily_output_path = os.path.join(
            data.cet_out['daily_output_ws'],
            data.cet_out['name_format'].replace(
                '%c', '%02d' % int(crop.class_number)) % et_cell.field_id)

        # Set output column order
        daily_output_columns = base_columns + [year_field, month_field,
                                               day_field, doy_field, pmet_field,
                                               etact_field, etpot_field,
                                               etbas_field, kc_field, kcb_field,
                                               precip_field, irrig_field,
                                               runoff_field, dperc_field,
                                               p_rz_field, p_eft_field,
                                               niwr_field, season_field]

        # Remove these (instead of appending) to preserve column order
        if not data.kc_flag:
            daily_output_columns.remove(kc_field)
            daily_output_columns.remove(kcb_field)
        if not data.niwr_flag:
            daily_output_columns.remove(niwr_field)

        # Most crops do not have cuttings, so append if needed
        if data.cutting_flag and crop.cutting_crop:
            daily_output_df[cutting_field] = daily_output_df[cutting_field].map(
                lambda x: ' %1d' % x)
            daily_output_columns.append(cutting_field)

        with open(daily_output_path, open_mode, newline='') as daily_output_f:
            daily_output_f.write('# {0:2d} - {1}\n'.format(
                crop.class_number, crop.name))
            daily_output_df.to_csv(
                daily_output_f, header=print_header, index=print_index,
                sep=',', columns=daily_output_columns,
                float_format=data.cet_out['daily_float_format'],
                date_format=data.cet_out['daily_date_format'])
        del daily_output_df, daily_output_path, daily_output_columns

    # Write monthly cet
    if data.cet_out['monthly_output_flag']:
        monthly_output_df[year_field] = monthly_output_df.index.year
        monthly_output_df[month_field] = monthly_output_df.index.month

        # format date attributes if values are formatted
        if data.cet_out['monthly_float_format'] is not None:
            monthly_output_df[year_field] = \
                monthly_output_df[year_field].map(lambda x: ' %4d' % x)
            monthly_output_df[month_field] = \
                monthly_output_df[month_field].map(lambda x: ' %2d' % x)
            monthly_output_df[season_field] = \
                monthly_output_df[season_field].map(lambda x: ' %2d' % x)
        monthly_output_path = os.path.join(
            data.cet_out['monthly_output_ws'],
            data.cet_out['name_format'].replace(
                '%c', '%02d' % int(crop.class_number)) % et_cell.field_id)
        monthly_output_columns = base_columns + [year_field, month_field,
                                                 pmet_field, etact_field,
                                                 etpot_field, etbas_field,
                                                 kc_field, kcb_field,
                                                 precip_field, irrig_field,
                                                 runoff_field, dperc_field,
                                                 p_rz_field, p_eft_field,
                                                 p_rz_fraction_field, p_eft_fraction_field,
                                                 niwr_field, season_field]
        if data.cutting_flag and crop.cutting_crop:
            monthly_output_df[cutting_field] = \
                monthly_output_df[cutting_field].map(lambda x: ' %1d' % x)
            monthly_output_columns.append(cutting_field)
        with open(monthly_output_path, open_mode, newline='') as monthly_output_f:
            monthly_output_f.write('# {0:2d} - {1}\n'.format(
                crop.class_number, crop.name))
            monthly_output_df.to_csv(
                monthly_output_f, header=print_header,
                index=print_index, sep=',', columns=monthly_output_columns,
                float_format=data.cet_out['monthly_float_format'],
                date_format=data.cet_out['monthly_date_format'])
        del monthly_output_df, monthly_output_path, monthly_output_columns

    # Write annual cet
    if data.cet_out['annual_output_flag']:
        annual_output_df[year_field] = annual_output_df.index.year
        annual_output_df[season_field] = annual_output_df[season_field].map(
            lambda x: ' %3d' % x)
        annual_output_path = os.path.join(
            data.cet_out['annual_output_ws'],
            data.cet_out['name_format'].replace(
                '%c', '%02d' % int(crop.class_number)) % et_cell.field_id)
        annual_output_columns = base_columns + [year_field, pmet_field,
                                                etact_field, etpot_field,
                                                etbas_field, kc_field,
                                                kcb_field, precip_field,
                                                irrig_field, runoff_field,
                                                dperc_field,
                                                p_rz_field, p_eft_field,
                                                p_rz_fraction_field, p_eft_fraction_field,
                                                niwr_field, season_field]
        try:
            annual_output_columns.remove('Date')
        except:
            pass
        if data.cutting_flag and crop.cutting_crop:
            annual_output_df[cutting_field] = annual_output_df[
                cutting_field].map(lambda x: ' %2d' % x)
            annual_output_columns.append(cutting_field)
        with open(annual_output_path, open_mode, newline='') as annual_output_f:
            annual_output_f.write('# {0:2d} - {1}\n'.format(
                crop.class_number, crop.name))
            annual_output_df.to_csv(
                annual_output_f, header=print_header,
                index=False, sep=',', columns=annual_output_columns,
                float_format=data.cet_out['annual_float_format'],
                date_format=data.cet_out['annual_date_format'])
        del annual_output_df, annual_output_path, annual_output_columns

    # Write growing season statistics
    if data.gs_output_flag:
        def doy_2_date(test_year, test_doy):
            try:
                return datetime.datetime.strptime(
                    '{0}_{1}'.format(int(test_year), int(
                        test_doy)), '%Y_%j').date().isoformat()
            except:
                return 'None'

        gs_output_df[gs_start_date_field] = \
            gs_output_df[[year_field, gs_start_doy_field]].apply(
                lambda s: doy_2_date(*s), axis=1)
        gs_output_df[gs_end_date_field] = gs_output_df[
            [year_field, gs_end_doy_field]].apply(
            lambda s: doy_2_date(*s), axis=1)
        if data.gs_name_format is None:
            # default filename spec
            gs_output_path = os.path.join(
                data.gs_output_ws, '{0}_gs_crop_{1:02d}.csv'.format(
                    et_cell.field_id, int(crop.class_number)))
        else:
            # user filename spec or function of cet name spec
            gs_output_path = os.path.join(
                data.gs_output_ws, data.gs_name_format.replace(
                    '%c', '%02d' % int(crop.class_number)) % et_cell.field_id)
        gs_output_columns = [
            year_field, gs_start_doy_field, gs_end_doy_field,
            gs_start_date_field, gs_end_date_field, gs_length_field]
        with open(gs_output_path, open_mode, newline='') as gs_output_f:
            gs_output_f.write(
                '# {0:2d} - {1}\n'.format(crop.class_number, crop.name))
            try:
                gs_start_doy = int(round(
                    gs_output_df[gs_start_doy_field].mean()))
            except:
                gs_start_doy = np.nan
            try:
                gs_end_doy = int(round(gs_output_df[gs_end_doy_field].mean()))
            except:
                gs_end_doy = np.nan
            if gs_start_doy is np.nan:
                logging.info('\nSkipping Growing Season Output for'
                             ' Cell ID: {} Crop: {:02d}'
                             .format(et_cell.field_id, int(crop.class_number)))
                return
            gs_start_dt = datetime.datetime.strptime(
                '2001_{:03d}'.format(gs_start_doy), '%Y_%j')
            gs_end_dt = datetime.datetime.strptime(
                '2001_{:03d}'.format(gs_end_doy), '%Y_%j')
            gs_output_f.write(
                '# Mean Start Date: {dt.month}/{dt.day}  ({doy})\n'.format(
                    dt=gs_start_dt, doy=gs_start_doy))
            gs_output_f.write(
                '# Mean End Date:   {dt.month}/{dt.day}  ({doy})\n'.format(
                    dt=gs_end_dt, doy=gs_end_doy))
            gs_output_df.to_csv(
                gs_output_f, sep=',', columns=gs_output_columns,
                date_format='%Y', index=False)
        del gs_output_df, gs_output_path, gs_output_columns


if __name__ == '__main__':
    pass
