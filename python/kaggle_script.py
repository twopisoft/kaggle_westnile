"""
Beating the Benchmark
West Nile Virus Prediction @ Kaggle
__author__ : Abhihsek
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import tune



def prep_test_data(train, test, weather):

    test['year'] = test.Date.apply(tune.create_year)
    test['month'] = test.Date.apply(tune.create_month)
    test['day'] = test.Date.apply(tune.create_day)

    test['Lat_int'] = test.Latitude.apply(int)
    test['Long_int'] = test.Longitude.apply(int)

    test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

    test = test.merge(weather, on='Date')
    test = test.drop(['Date'], axis = 1)

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train['Species'].values) + list(test['Species'].values))
    train['Species'] = lbl.transform(train['Species'].values)
    test['Species'] = lbl.transform(test['Species'].values)

    lbl.fit(list(train['Street'].values) + list(test['Street'].values))
    train['Street'] = lbl.transform(train['Street'].values)
    test['Street'] = lbl.transform(test['Street'].values)

    lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
    train['Trap'] = lbl.transform(train['Trap'].values)
    test['Trap'] = lbl.transform(test['Trap'].values)

    test = test.ix[:,(test != -1).any(axis=0)]

    return test

def main():

    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    weather_df = pd.read_csv('../input/weather.csv')

    weather = tune.prep_weather_data(weather_df)

    train = tune.prep_train_data(train_df, test_df, weather)

    test = prep_test_data(train, test_df, weather)

    sample = pd.read_csv('../input/sampleSubmission.csv')

    # GBM classifier
    clf = ensemble.GradientBoostingClassifier(n_estimators=100, min_samples_split=1)

    clf = tune.train_classifier(clf, train, ['2007','2009','2011','2013'])

    # create predictions and submission file
    predictions = clf.predict_proba(test)[:,1]
    sample['WnvPresent'] = predictions
    sample.to_csv('../output/gbm_100_2.csv', index=False)

if __name__ == "__main__":
    main()
