import numpy as np
import pandas as pd
from gridding import Gridder
from sklearn.preprocessing import LabelBinarizer
from joblib import load

def check_active_fires_lon_cols(dfr):
    if 'frp' not in dfr.columns:
        dfr = add_frp(dfr)
    if 'longitude' not in dfr.columns:
        gri = Gridder(step = 0.25, bbox = 'indonesia')
        dfr = gri.add_coords_from_ind(dfr)
    return dfr

class Reliability(object):
    def __init__(self, forecasts, observations, thresholds, obs_threshold):
        self.forecasts = forecasts
        self.observations = observations
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        self.pos_relative_frequency = np.zeros(self.thresholds.shape)
        self.total_relative_frequency = np.zeros(self.thresholds.shape)
        self.calc_reliability_curve()

    def calc_reliability_curve(self):
        pos_frequency = np.zeros(self.thresholds.shape)
        total_frequency = np.zeros(self.thresholds.shape)
        for t, threshold in enumerate(self.thresholds[:-1]):
            pos_frequency[t] = np.count_nonzero((threshold <= self.forecasts) &
                                                (self.forecasts < self.thresholds[t+1]) &
                                                (self.observations > self.obs_threshold))
            total_frequency[t] = np.count_nonzero((threshold <= self.forecasts) &
                                                  (self.forecasts < self.thresholds[t+1]))
            if total_frequency[t] > 0:
                self.pos_relative_frequency[t] = pos_frequency[t] / float(total_frequency[t])
                self.total_relative_frequency[t] = total_frequency[t] / self.forecasts.size
            else:
                self.pos_relative_frequency[t] = np.nan
        #self.pos_relative_frequency[-1] = np.nan

    def brier_score(self):
        obs_truth = np.where(self.observations >= self.obs_threshold, 1, 0)
        return np.mean((self.forecasts - obs_truth) ** 2)

    def brier_score_components(self):
        obs_truth = np.where(self.observations >= self.obs_threshold, 1, 0)
        #obs_truth = labels
        climo_freq = obs_truth.sum() / float(obs_truth.size)
        total_freq = self.total_relative_frequency[:-1] * self.forecasts.size
        bins = 0.5 * (self.thresholds[0:-1] + self.thresholds[1:])
        pos_rel_freq = np.where(np.isnan(self.pos_relative_frequency[:-1]), 0, self.pos_relative_frequency[:-1])
        reliability = np.sum(total_freq * (bins - pos_rel_freq) ** 2)/ len(self.observations)
        resolution = np.sum(total_freq * (pos_rel_freq - climo_freq) ** 2) / len(self.observations)
        uncertainty = climo_freq * (1 - climo_freq)
        return reliability, resolution, uncertainty

    def brier_skill_score(self):
        obs_truth = np.where(self.observations >= self.obs_threshold, 1, 0)
        climo_freq = obs_truth.sum() / float(obs_truth.size)
        bs_climo = np.mean((climo_freq - obs_truth) ** 2)
        bs = self.brier_score()
        return 1.0 - bs / bs_climo

    def __str__(self):
        return "Brier Score: {0:0.3f}, Reliability: {1:0.3f}, Resolution: {2:0.3f}, Uncertainty: {3:0.3f}".format(
            tuple([self.brier_score()] + list(self.brier_score_components())))

def feature_relevance(clf, test, x_test_scaled, y_test, feats):
    lr = LRP(clf)
    y_pred, relevancies = lr.get_props(x_test_scaled, y_test, 0)
    col_names = ['rel_{0}'.format(x) for x in feats]
    rel_dfr = pd.DataFrame(relevancies, columns = col_names)
    return pd.concat([test.reset_index(), rel_dfr.reset_index()], axis = 1)

def add_feature_relevance(dfr, feats, clf_name):
    dfrs = []
    for year in dfr.year.unique():
        dfs = dfr[dfr.year == year].copy()
        #clf = load('data/clfs/clf_{0}_{1}.joblib'.format(clf_name, year))
        clf = load('data/clf_{0}.joblib'.format(clf_name))
        #scaler = load('data/clfs/scaler_{0}_{1}.joblib'.format(clf_name, year))
        scaler = load('data/scaler_{0}.joblib'.format(clf_name))
        x_test_scaled = scaler.transform(dfr[dfr.year == year][feats])
        dfs = feature_relevance(clf, dfs, x_test_scaled, dfs.labels, feats)
        dfrs.append(dfs)
    return ps.concat(dfrs)

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

def add_fire_clim(dfr):
    clim_probs = (dfr.groupby(['lonind', 'latind', 'month'])['labels'].value_counts().unstack() / 18.).fillna(0).reset_index()
    clim_probs = clim_probs.rename({0.0: 'cp_0', 1.0: 'cp_1', 2.0: 'cp_2'}, axis = 1)
    dfr = pd.merge(dfr, clim_probs, on = ['lonind', 'latind', 'month'], how = 'left')
    return dfr

def pod(probs, labels):
    """
    Probability of detection
    Args:
        probs (Array): array with predicted probabilities
        labels (Array): array with true labels
    Returns:
        pods (Array): array with probability of detection
        for a range of probability thresholds [0, 100]
    """
    pods = []
    for thr in np.linspace(0, 1, 101):
        tps = probs[(probs > thr) & (labels == 1)].shape[0]
        fns = probs[(probs < thr) & (labels == 1)].shape[0]
        try:
            pod = tps / (tps + fns)
        except ZeroDivisionError:
            pod = np.nan
        pods.append(pod)
    return np.array(pods)

def pofd(probs, labels):
    pofds = []
    for thr in np.linspace(0, 1, 101):
        fps = probs[(probs > thr) & (labels == 0)].shape[0]
        tns = probs[(probs < thr) & (labels == 0)].shape[0]
        try:
            pofd = fps / (fps + tns)
        except ZeroDivisionError:
            pofd = 0
        pofds.append(pofd)
    return pofds

def labels_binarize(labels):
    ll = LabelBinarizer()
    ll.fit(labels)
    return ll.transform(labels)

def rps(probs, labels):
    labels_b = labels_binarize(labels)
    num_class = labels_b.shape[1]
    cum_outcomes = np.cumsum(labels_b, axis = 1)
    cum_probs = probs.cumsum(axis = 1)
    sum_rps = np.zeros_like(labels)
    for i in range(num_class):         
        sum_rps+= (cum_probs.iloc[:, i] - cum_outcomes[:, i])**2
    return sum_rps/(num_class - 1)

def compute_rps(dfr):
    for item in ['era5', 'clim_1', 'clim_2', 'clim_3', 'clim_4', 'clim_5', 'clim_6',
            'fore_1', 'fore_2', 'fore_3', 'fore_4', 'fore_5', 'fore_6']:
        dfr[item + '_rps'] = rps(dfr.filter(like = item), dfr.labels)
    return dfr

def rpss(dfr):
    for lead in range(1, 7):
        dfr['rpss_{}'.format(lead)] = 1 - (dfr['fore_{}_rps'.format(lead)] /
                                           dfr['clim_{}_rps'.format(lead)])
    return dfr

def rel_value(dfr, col, label, prob_thr):
    hits = dfr[(dfr[col] > prob_thr) & (dfr.labels == label)].shape[0] / dfr.shape[0]
    false_alarms = dfr[(dfr[col] > prob_thr) & (dfr.labels < label)].shape[0] / dfr.shape[0]
    misses = dfr[(dfr[col] < prob_thr) & (dfr.labels == label)].shape[0] / dfr.shape[0]
    if label == 1:
        pclim = (dfr['cp_1'] + dfr['cp_2']).mean()
    if label == 2:
        pclim = dfr['cp_2'].mean()
    vs = []
    for cl in np.linspace(0.01, 0.99, 99):
        if cl < pclim:
            value = (cl * (hits + false_alarms - 1) + misses) / (cl * (pclim - 1))
        else:
            value = (cl * (hits + false_alarms) + misses - pclim) / (pclim * (cl - 1))
        if value < 0:
            value = np.nan
        vs.append(value)
    return vs

