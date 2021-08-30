"""
@author: Tadas Nikonovas
helper functions, part of:
ProbFire, a probabilistic fire early warning system for
Indonesia.
"""
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from gridding import *

def prepare_dfr(dfr):
    """
    Make sure the dfr dataframe has all needed columns
    """
    dfr = check_active_fires_lon_cols(dfr)
    dfr = add_frp_acc(dfr)
    dfr = add_region_labels(dfr)
    return dfr

def add_region_labels(dfr):
    from utils.gridding import bboxes
    dfrs = dfr.copy()
    dfrs['region'] = 0
    dfrs = check_active_fires_lon_cols(dfrs)
    for nr, (key, value) in enumerate(bboxes.items(), 1):
        dfrb = spatial_subset_dfr(dfrs, value).copy()
        dfrs.loc[dfrs.index.isin(dfrb.index), 'region'] = nr
    ll = LabelBinarizer()
    ll.fit(dfrs.region)
    labs =  ll.transform(dfrs.region)
    labs = pd.DataFrame(labs, columns = ['nn', 'sk', 'ss', 'sp', 'cs', 'wk'])
    return pd.concat([dfrs, labs], axis = 1)

def add_frp_acc(frp):
    #prepare atsr night time fire data
    atsr = pd.read_parquet('/home/tadas/pyviz/data/atsr_indo_1997_2002.parquet')
    #the below grid cells contain "stationary" active fire pixels, hence are removed
    atsr.loc[((atsr.lonind == 56) & (atsr.latind == 27)), 'frp'] = 0
    atsr.loc[((atsr.lonind == 76) & (atsr.latind == 24)), 'frp'] = 0
    atsr.loc[((atsr.lonind == 98) & (atsr.latind == 54)), 'frp'] = 0
    atsr.loc[((atsr.lonind == 98) & (atsr.latind == 55)), 'frp'] = 0
    atsr.loc[((atsr.lonind == 118) & (atsr.latind == 36)), 'frp'] = 0
    atsr.loc[((atsr.lonind == 113) & (atsr.latind == 42)), 'frp'] = 0
    atsr.loc[((atsr.lonind == 114) & (atsr.latind == 42)), 'frp'] = 0
    gri = Gridder(step = 0.25, bbox = 'indonesia')
    atsr = gri.add_grid_inds(atsr)
    atsrg = atsr.groupby(['lonind', 'latind', 'year'])['fire'].sum().reset_index()
    atsrm = atsrg.groupby(['lonind', 'latind'], as_index=False)['fire'].max()
    atsrm['year'] = 2001
    atsrm = atsrm.rename({'fire': 'frp'}, axis = 1)
    frpg = pd.concat([atsrm, frp[['lonind', 'latind', 'year', 'frp']]])
    frpg = frpg.groupby(['lonind', 'latind', 'year'])['frp'].sum()
    frpm = frpg.groupby(level = [0, 1]).cummax().reset_index()
    frpm = frpm.rename({'frp': 'frp_max'}, axis = 1)
    frpm['year'] += 1
    frp = frp.merge(frpm, on = ['lonind', 'latind', 'year'], how = 'left')

    #fire counts in prev years
    atsrg = atsr.groupby(['lonind', 'latind', 'year'])['fire'].max().reset_index()
    atsrs = atsrg.groupby(['lonind', 'latind'], as_index=False)['fire'].sum()
    atsrs['year'] = 2001
    atsrs = atsrs.rename({'fire': 'count'}, axis = 1)
    frpc = frp[frp.year != 0][['lonind', 'latind', 'year', 'frp']].copy()
    frpc['count'] = 0
    frpc.loc[frpc.frp > 10, 'count'] = 1
    frpg = pd.concat([atsrs, frpc[['lonind', 'latind', 'year', 'count']]])
    cs = frpg.groupby(['lonind', 'latind', 'year'])['count'].max().groupby(level = [0, 1]).cumsum().reset_index()
    cs['year'] += 1
    cs = cs.rename({'count': 'frp_count'}, axis = 1)
    frp = frp.merge(cs, on = ['lonind', 'latind', 'year'], how = 'left')
    frp = frp.fillna(0)
    return frp

def check_active_fires_lon_cols(dfr):
    if 'frp' not in dfr.columns:
        dfr = add_frp(dfr)
    if 'longitude' not in dfr.columns:
        gri = Gridder(step = 0.25, bbox = 'indonesia')
        dfr = gri.add_coords_from_ind(dfr)
    return dfr

def add_frp(dfr):
    frp = pd.read_parquet('data/feature_train_fr_0.25deg_2019_12_31.parquet')
    frp.loc[((frp.lonind == 56) & (frp.latind == 27)), 'frp'] = 0
    frp.loc[((frp.lonind == 76) & (frp.latind == 24)), 'frp'] = 0
    frp.loc[((frp.lonind == 98) & (frp.latind == 54)), 'frp'] = 0
    frp.loc[((frp.lonind == 98) & (frp.latind == 55)), 'frp'] = 0
    frp.loc[((frp.lonind == 118) & (frp.latind == 36)), 'frp'] = 0
    frp.loc[((frp.lonind == 113) & (frp.latind == 42)), 'frp'] = 0
    frp.loc[((frp.lonind == 114) & (frp.latind == 42)), 'frp'] = 0
    dfr = pd.merge(dfr, frp[['lonind', 'latind', 'year', 'month', 'frp']],
                   on = ['lonind', 'latind', 'year', 'month'], how = 'left')
    dfr = dfr.fillna(0)
    return dfr

def forecast_all(clf_name, clabel):
    dfrs = []
    for lead in range(1, 7):
        fc = pd.read_parquet('data/{0}_{1}.parquet'.format(clf_name, lead))
        name = 'fore_{}_'.format(lead)
        cname = '_{}'.format(clabel)
        regstring = name + '(\d{1,2})' + cname
        f_cols = fc.filter(regex = regstring)
        dfrs.append(f_cols.reset_index(drop = True))
    return pd.concat(dfrs, axis = 1)


def forecast_sumary(clf_name):
    dc = {}
    for lead in range(1, 7):
        fc = pd.read_parquet('data/{0}_{1}.parquet'.format(clf_name, lead))
        for clabel in [0, 1, 2]:
            name = 'fore_{}_'.format(lead)
            cname = '_{}'.format(clabel)
            regstring = name + '(\d{1,2})' + cname
            f_cols = fc.filter(regex = regstring)
            meds = f_cols.mean(axis = 1)
            q75 = f_cols.quantile(.75, axis = 1)
            q25 = f_cols.quantile(.25, axis = 1)
            dc['fore_{0}_{1}'.format(lead, clabel)] = meds
            dc['fore_{0}_{1}_q75'.format(lead, clabel)] = q75
            dc['fore_{0}_{1}_q25'.format(lead, clabel)] = q25
    dfr = pd.DataFrame(dc)
    return dfr


