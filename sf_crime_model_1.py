import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss


def base_feature(df):
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['dayofweek'] = df['Dates'].dt.dayofweek
    df['hour'] = df['Dates'].dt.hour
    df['Corner'] = df['Address'].map(lambda x: '/' in x)
    return df

def check_model(X,y):
    #use stratified shuffle for minority class
    sss = StratifiedShuffleSplit(y,test_size=0.3,n_iter=2)
    for train_index, test_index in sss:
        X_train, X_cv = X.ix[train_index], X.ix[test_index]
        y_train, y_cv = y.ix[train_index], y.ix[test_index]
    #check max_depth
    for md in (11,12,13,14,15):
        rf = RandomForestClassifier(max_depth=md)
        rf.fit(X_train,y_train)
        ll = log_loss(y_cv,rf.predict_proba(X_cv))
        print 'Max depth: {0}, Log loss: {1}'.format(md,ll)
    #check max_features
    for mf in (1,2,3,4,5,6):
        rf = RandomForestClassifier(max_depth=13,max_features=mf)
        rf.fit(X_train,y_train)
        ll = log_loss(y_cv,rf.predict_proba(X_cv))
        print 'Max features: {0}, Log loss: {1}'.format(mf,ll)
    #check num_estimators
    for ne in (10,20,30,40,50):
        rf = RandomForestClassifier(max_depth=13,max_features=4,n_estimators=ne)
        rf.fit(X_train,y_train)
        ll = log_loss(y_cv,rf.predict_proba(X_cv))
        print 'N estimators: {0}, Log loss: {1}'.format(ne,ll)

def make_submission(X,y,X_test):
    rf = RandomForestClassifier(max_depth=13,max_features=4,n_estimators=50)
    rf.fit(X,y)
    y_pred = rf.predict_proba(X_test)
    classes = rf.classes_
    y_test = pd.DataFrame({classes[i]:y_pred[:,i] for i in range(len(classes))})
    y_test['Id'] = test['Id']
    y_test[['Id']+list(classes)].to_csv('submission.csv',index=False)


if __name__ =='__main__':
    train = pd.read_csv('train.csv',parse_dates=['Dates'])
    train = base_feature(train)
    test = pd.read_csv('test.csv',parse_dates=['Dates'])
    test = base_feature(test)

    features = ['PdDistrict', 'Address', 'Corner', 'year', 'month', 'day', 'dayofweek','hour']

    X = train[features]
    y = train['Category']
    X_test = test[features]
    #use labelencoder() to encode PdDistrict and Address for both train and test
    for cat in ['PdDistrict','Address']:
        le = LabelEncoder()
        unq_vals = pd.concat([X[cat],X_test[cat]]).unique()
        le.fit(unq_vals)
        X[cat] = le.transform(X[cat])
        X_test[cat] = le.transform(X_test[cat])

    check_model(X,y)
    make_submission(X,y,X_test)
