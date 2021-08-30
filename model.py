"""
@author: Tadas Nikonovas
probabilistic classifier model based on sklearn MLPClassifier,
part of: ProbFire, a probabilistic fire early warning system for
Indonesia.
"""
import os, copy
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from utils.utils import *
from utils.gridding import *

def forecast_full(clf_name, clabel):
    dfrs = []
    for lead in range(1, 7):
        fc = pd.read_parquet('data/{0}_{1}.parquet'.format(clf_name, lead))
        name = 'fore_{}_'.format(lead)
        cname = '_{}'.format(clabel)
        regstring = name + '(\d{1,2})' + cname
        f_cols = fc.filter(regex = regstring)
        dfrs.append(f_cols.reset_index(drop = True))
    return pd.concat(dfrs, axis = 1)

def read_features(name, lead = None, number = None):
    data_path = '/mnt/data2/forecast/features/merged/'
    if lead and number is not None:
        fname = '{0}_{1}_{2}.parquet'.format(name, lead, number)
    elif lead:
        fname = '{0}_{1}.parquet'.format(name, lead)
    else:
        fname = '{}.parquet'.format(name)
    dfr = pd.read_parquet(os.path.join(data_path, fname))
    print(dfr.columns)
    dfr['peat'] = dfr.peatd
    return dfr

def get_year_train_test(dfr, year):
    train = dfr[dfr.year != year]
    test = dfr[dfr.year == year]
    return train, test

def class_labels(dfr):
    labels = np.zeros_like(dfr.frp.values)
    labels[(dfr['frp'] > 0) & (dfr['frp'] < 11)] = 1
    labels[(dfr['frp'] > 10)] = 2
    return labels

def labels_binarize(labels):
    ll = LabelBinarizer()
    ll.fit(labels)
    return ll.transform(labels)

class ProbFire():
    def __init__(self, features, model_name):
        self.clf = MLPClassifier(solver='lbfgs', alpha = 1,
                                 hidden_layer_sizes=(15),
                                 max_iter = 50000,
                                 activation='relu', random_state=1)
        self.clfs = {}
        self.features = features
        self.name = model_name
        self.years = range(2002, 2020, 1)

    def fit(self, X, labels):
        self.unique_class = np.sort(np.unique(labels))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (labels > self.unique_class[i]).astype(np.uint8)
                clf = copy.copy(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                 predicted.append(clfs_predict[y-1][:,1])
            else:
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T

    def train_year_models(self, dataset)
        for year in self.years:
            train, test = get_year_train_test(dataset, year)
            x_train = train[self.features]
            x_test = test[self.features]
            scaler = preprocessing.StandardScaler().fit(x_train.values)
            x_train_scaled = scaler.transform(x_train.values)
            x_test_scaled = scaler.transform(x_test.values)
            self.clf.fit(x_train_scaled, y_train)
            dump(self.clf, 'data/clf_{0}_{1}.joblib'.format(self.name, year))
            dump(scaler, 'data/scaler_{0}_{1}.joblib'.format(self.name, year))

    def predict_years(self, dataset, columns_name):
        predicted = []
        for year in self.years:
            clf = load('data/clf_{0}_{1}.joblib'.format(self.name, year))
            scaler = load('data/scaler_{0}_{1}.joblib'.format(self.name, year))
            test = dfr[dfr.year == year].copy()
            x_test = test[self.features]
            x_test_scaled = scaler.transform(x_test.values)
            probas = clf.predict_proba(x_test_scaled)
            df = pd.DataFrame({'{}_1'.format(columns_name): probas[:, 1] + probas[:, 2],
                               '{}_2'.format(columns_name): probas[:, 2]})
            df.index = test.index
            predicted.append(df)
        predicted = pd.concat(predicted)
        return predicted


#instantianate geospatial hellper object for the region
gri = Gridder(bbox = 'indonesia', step = 0.25)

#define feature column names
climate_features = ['tp', 'tp_1', 'tp_2', 'tp_3', 'tp_4', 'tp_5',
                    't2m', 'h2m']
non_climate_features = ['loss_last_prim', 'loss_last_sec',
                        'frp_max', 'f_prim', 'peat',
                        'cs', 'ss', 'wk', 'sk', 'sp']
features = climate_features + non_climate_features

fm = ProbFire(features, model_name = 'base')
era_dataset = pd.read_parquet('data/era5_features.parquet')
era_dataset['labels'] = class_labels(era_dataset)
era_dataset['labels'] = era_dataset['labels'].astype(int)

#train models for each year in the record
fm.train_year_models(era_dataset)

#predict class probabilities for era5 features
era_predictions = fm.predict_years(era_dataset, 'era5')
#add the predicted probabilites to longitude, latitude and date columns 
results = pd.concat([era_dataset[['lonind', 'latind', 'year', 'month', 'labels']],
                     era_predictions], axis = 1)

#predict probablities based on era5 climatology features 
leads = []
#do this for lead times 1 through 6
for lead in range(1, 7):
    dfr = read_features('clim', lead)
    dfr = prepare_dfr(dfr)
    predicted = fm.predict_years(dfr, 'clim_{}'.format(lead))
    leads.append(predicted)
clim_predictions = pd.concat(leads, axis = 1)
#add climatology predictions to the results dataframe
results = pd.concat([results, clim_predictions], axis = 1)

#predict probabilities based on SEAS5 features
#do this for lead times 1 through 6
for lead in range(1, 7):
    members = []
    #and for SEAS5 ensemble member 1 - 25
    for number in range(0, 25):
        dfr = read_features('fore_corv2', lead = lead, number = number)
        dfr = prepare_dfr(dfr)
        predicted = fm.predict_years(dfr, 'fore_{0}_{1}'.format(lead, number))
        members.append(predicted)
    lead_predictions = pd.concat(members, axis = 1)
    lead_results = pd.concat([results, lead_predictions], axis = 1)
    lead_predictions.to_parquet('data/{0}_{1}.parquet'.format(fm.name, lead))

#the bellow compiles all results (era5, climatology and SEAS5) for
#fire classes 1 and 2
for label in [1, 2]:
    predictions = forecast_full(fm.name, label)
    label_predictions = pd.concat([results, predictions], axis = 1)
    #check and add back longitude and latitude columns
    label_predictions = check_active_fires_lon_cols(label_predictions)
    #write to parquet file
    label_precictions.to_parquet('data/results_{0}_{1}.parquet'.format(fm.name, label))
