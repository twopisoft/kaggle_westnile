"""
Beating the Benchmark
West Nile Virus Prediction @ Kaggle
__author__ : Abhihsek
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, cross_validation
from sklearn import metrics

def train_classifier(clf, train, y):
    y_pred = np.zeros((len(y),len(set(y))))
    kf = cross_validation.StratifiedKFold(y, n_folds=5)
    for trn, tst in kf:
        X_train, X_test, y_train, y_test = train.ix[trn,:], train.ix[tst,:], y[trn], y[tst]
        clf.fit(X_train, y_train)
        y_pred[tst,:] = clf.predict_proba(X_test)
    return clf

def roc_area(y_true, y_pred):
    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred[:,1])
    return metrics.auc(fpr, tpr)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

def prep_data():
    # Load dataset 
    train = pd.read_csv('../input/train.csv')
    weather = pd.read_csv('../input/weather.csv')

    # Get labels
    #labels = train.WnvPresent.values

    # Not using codesum for this benchmark
    weather = weather.drop('CodeSum', axis=1)

    # Split station 1 and 2 and join horizontally
    weather_stn1 = weather[weather['Station']==1]
    weather_stn2 = weather[weather['Station']==2]
    weather_stn1 = weather_stn1.drop('Station', axis=1)
    weather_stn2 = weather_stn2.drop('Station', axis=1)
    weather = weather_stn1.merge(weather_stn2, on='Date')

    # replace some missing values and T with -1
    weather = weather.replace('M', -1)
    weather = weather.replace('-', -1)
    weather = weather.replace('T', -1)
    weather = weather.replace(' T', -1)
    weather = weather.replace('  T', -1)

    train['month'] = train.Date.apply(create_month)
    train['day'] = train.Date.apply(create_day)

    # Add integer latitude/longitude columns
    train['Lat_int'] = train.Latitude.apply(int)
    train['Long_int'] = train.Longitude.apply(int)

    # drop address columns
    train = train.drop(['Address', 'AddressNumberAndStreet', 'NumMosquitos'], axis = 1)

    # Merge with weather data
    train = train.merge(weather, on='Date')
    train = train.drop(['Date'], axis = 1)

    # Convert categorical data to numbers
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train['Species'].values))
    train['Species'] = lbl.transform(train['Species'].values)

    lbl.fit(list(train['Street'].values))
    train['Street'] = lbl.transform(train['Street'].values)

    lbl.fit(list(train['Trap'].values))
    train['Trap'] = lbl.transform(train['Trap'].values)

    # drop columns with -1s
    train = train.ix[:,(train != -1).any(axis=0)]

    return train

def train_test_data(train):
    return cross_validation.train_test_split(train,train_size=0.7,random_state=1)

def main():
    data = prep_data()
    train,val = train_test_data(data)

    labels = train.WnvPresent.values
    train = train.drop(['WnvPresent'], axis = 1)

    # GBM classifier
    clf = ensemble.GradientBoostingClassifier(n_estimators=1000, min_samples_split=1)
    clf.fit(train, labels)

    y = val.WnvPresent.values
    val = val.drop(['WnvPresent'], axis = 1)
    y_pred = clf.predict_proba(val)
    print "ROC area = {}".format(roc_area(y, y_pred))

if __name__ == "__main__":
    main()



