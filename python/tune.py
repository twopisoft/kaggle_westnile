import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, cross_validation
from sklearn import metrics

def roc_area(y_true, y_pred):
    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred[:,1])
    return metrics.auc(fpr, tpr)

def create_year(x):
    return x.split('-')[0]

def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

def prep_weather_data(weather):

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

    return weather

def prep_train_data(train, test, weather):

    train['year'] = train.Date.apply(create_year)
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

    # drop columns with -1s
    train = train.ix[:,(train != -1).any(axis=0)]

    return train

def train_test_data(train):
    return cross_validation.train_test_split(train,train_size=0.7,random_state=1)

def train_test_data_year(train,trn_years,tst_years):
    return train[ train['year'].isin(trn_years) ], train[ train['year'].isin(tst_years) ]

def rotate(l,n):
    return l[n:] + l[:n]

def train_classifier(clf, data, years):
    
    avg_roc = 0.0

    for i in range(len(years)):
    #for i in range(1):
        yrs = rotate(years,i)
        trn_years = yrs[:-1]
        tst_years = yrs[-1:]

        train,val = train_test_data_year(data,trn_years,tst_years)

        labels = train.WnvPresent.values
        train = train.drop(['WnvPresent'], axis = 1)

        clf.fit(train, labels)

        y = val.WnvPresent.values
        val = val.drop(['WnvPresent'], axis = 1)

        y_pred = clf.predict_proba(val)

        roc = roc_area(y, y_pred)
        avg_roc += roc
        print "Train Years: {0}, Test Years = {1}, ROC area = {2}".format(trn_years,tst_years,roc)

    print "Average ROC = {}".format(avg_roc/len(years))

    return clf

def main():

    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    weather_df = pd.read_csv('../input/weather.csv')

    weather = prep_weather_data(weather_df)

    data = prep_train_data(train_df, test_df, weather)

    # GBM classifier
    clf = ensemble.GradientBoostingClassifier(loss='exponential', n_estimators=100, min_samples_split=1)

    train_classifier(clf, data, ['2007','2009','2011','2013'])

if __name__ == "__main__":
    main()



