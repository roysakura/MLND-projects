import warnings
warnings.filterwarnings('ignore')

from IPython.display import display, HTML
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import math

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC,SVR
from sklearn.metrics import r2_score
import xgboost as xgb
from xgboost import XGBRegressor 

# Preprocessing
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn import preprocessing
# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

#Imputation
#from fancyimpute import BiScaler, KNN

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

#Create some helpers functions

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def metrics_rmspe(y,yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_score(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

#Load the data from csv
store = pd.read_csv('./input/store.csv')
store_state = pd.read_csv('./input/store_states.csv')
train = pd.read_csv('./input/train.csv', parse_dates=["Date"],keep_date_col=True)
test = pd.read_csv('./input/test.csv', parse_dates=["Date"],keep_date_col=True)

#Data information Analysis including the data type and how many observations etc.
store = pd.merge(store,store_state,on='Store')
display(store.info())
display(train.info())

# Build features
def build_features(features):
    features['PromoInterval'].fillna('n',inplace=True)
    features['StateHoliday'] = features['StateHoliday'].replace('a',1)
    features['StateHoliday'] = features['StateHoliday'].replace('b',2)
    features['StateHoliday'] = features['StateHoliday'].replace('c',3)
    features['StateHoliday'] = features['StateHoliday'].astype(float)
    AssortStore = {'aa':1,'ab':2,'ac':3,'ad':4,'ba':5,'bb':6,'bc':7,'bd':8,'ca':9,'cb':10,'cc':11,'cd':12}
    features['AssortStore'] = features['Assortment']+features['StoreType']
    features['AssortStore'] = features['AssortStore'].map(AssortStore)
    features['Date'] = pd.to_datetime(features['Date'])
    States = {'BE':1,'BW':2,'BY':3,'HB,NI':4,'HE':5,'HH':6,'NW':7,'RP':8,'SH':9,'SN':10,'ST':11,'TH':12}
    features['State'] = features['State'].map(States)
    
    features['DayOfWeekPlusState'] = features['DayOfWeek'].astype(float)+features['StateHoliday']
    features['DayOfWeekPlusSchool'] = features['DayOfWeek'].astype(float)+features['SchoolHoliday']
    features['DayOfPromo'] = features['DayOfWeek'].astype(float)+(features['Promo'].astype(float)/2.0)
    features['WeekOfYear'] = features['Date'].map(lambda x: x.isocalendar()[1]).astype(float)
    features['DayOfYear'] = features['Date'].map(lambda x: x.timetuple().tm_yday).astype(float)
    features['Day']=features['Date'].map(lambda x:x.day).astype(float)
    features['Month'] = features['Date'].map(lambda x: x.month).astype(float)
    features['Year'] = features['Date'].map(lambda x: x.year).astype(float)
    features['Season'] = features['Date'].map(lambda x: 1 if x.month in [1,2,3] else (2 if x.month in [4,5,6] else (3 if x.month in [7,8,9] else 4))).astype(float)
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    features['monthStr'] = features.Month.map(month2str)
    features.loc[features.PromoInterval == 0, 'PromoInterval'] = ''
    features['IsPromoMonth'] = 0
    for interval in features.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                features.loc[(features.monthStr == month) & (features.PromoInterval == interval), 'IsPromoMonth'] = 1.0
    
    features["CompetitionDistance"].fillna(0,inplace=True)
    features["CompetitionDistance"] = np.log(features["CompetitionDistance"]+1)
    features["CompetitionMonthDuration"] = (features['Month']-features['CompetitionOpenSinceMonth'])+(features['Year']-features['CompetitionOpenSinceYear'])*12
    features["Promo2WeekDuration"] = (features['WeekOfYear']-features['Promo2SinceWeek'])/4+(features['Year']-features['Promo2SinceYear'])*12
    features['Promo2WeekDuration'] = features.Promo2WeekDuration.apply(lambda x: x if x > 0 else 0)
    features.loc[features.Promo2SinceYear == 0, 'Promo2WeekDuration'] = 0
    
    features["CompetitionMonthDuration"].fillna(0,inplace=True)
    
    features["Promo2WeekDuration"].fillna(0,inplace=True)
    features["CompetitionMonthDuration"] = np.log(features["CompetitionMonthDuration"]+1)
    features["Promo2WeekDuration"] = (features["Promo2WeekDuration"]+1)
    PromoCombo = {'11':1,'10':2,'01':3,'00':4}
    features['PromoCombo'] = features['Promo'].astype(str)+features['Promo2'].astype(str)
    features['PromoCombo'] = features['PromoCombo'].map(PromoCombo)
    
    features = features.drop(['Date','PromoInterval','Promo2SinceWeek','Promo2SinceYear','CompetitionOpenSinceYear','CompetitionOpenSinceMonth','monthStr','Season','Assortment','StoreType','StateHoliday','Promo2','SchoolHoliday','IsPromoMonth'],axis=1)
    if 'Customers' in features.columns:
        features = features.drop(['Customers'],axis=1)
    features.fillna(0,inplace=True)
    
    return features

feature_building_start = time.time()

# Caculate Sales Per Store Value
import warnings
warnings.filterwarnings('ignore')
store['SalesPerStore'] = np.zeros(store.Store.shape)
for i in range(1,len(store)+1):
    avg = (train[train.Store==i][train.Sales>0]['Sales']/train[train.Store==i][train.Sales>0]['Customers']).mean()
    store.set_value(store.Store==i,"SalesPerStore",avg)

org_train_data = pd.merge(train,store,on='Store')
org_train_data = org_train_data[org_train_data['Sales']>0]
org_train_data.set_index(['Date','Store'],inplace=True,drop=False)
org_train_data.sort_index(inplace=True)

org_test_data = pd.merge(test,store,on='Store')
org_test_data.set_index(['Date','Store'],inplace=True,drop=False)
org_test_data.sort_index(inplace=True)


    
combine_data = org_train_data.copy()
combine_data = org_train_data.append(org_test_data)

combine_data = combine_data.copy()


new_combine_data = pd.DataFrame([])
min_start = 43
max_end = 44

# Split the train data and test data
from datetime import datetime
test_start_date = '2015-08-01'
test_start_date = datetime.strptime(test_start_date,'%Y-%m-%d')

org_train_data = combine_data[combine_data.Date<test_start_date].drop(['Id'],axis=1)
org_test_data = combine_data[combine_data.Date>=test_start_date]

# Strat first subsample training
gbm_list = []
sub_range = 10
for s in range(1,sub_range+1):
    small_sample = range(s,len(store)+1,sub_range)

    org_train_sum_data = org_train_data[org_train_data.Store.isin(small_sample)]

    val_start_date = org_train_sum_data.iloc[-1].Date - timedelta(weeks=2)
    mask = (org_train_sum_data['Date'] >= val_start_date) & (org_train_sum_data['Date'] <= org_train_sum_data.iloc[-1].Date)

    val_data = org_train_sum_data.loc[mask]
    train_data = org_train_sum_data.loc[~mask]
    train_data = build_features(train_data)
    val_data = build_features(val_data)
    
    xgboost_start = time.time()

    features = [x for x in train_data.columns if x not in ['Sales']]

    dtrain = xgb.DMatrix(train_data[features], np.log(train_data["Sales"] + 1))
    dvalid = xgb.DMatrix(val_data[features], np.log(val_data["Sales"] + 1))
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    params = {"objective": "reg:linear",
              "booster" : "gbtree",
              "eta": 0.01,
              "max_depth": 10,
              "subsample": 0.3,
              "colsample_bytree": 0.8,
              "min_child_weight":8,
              "reg_alpha":1e-04,
              "seed":3131,
              "silent": 1
              }
    
    num_trees = 30000
    
    res = {}
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=350, feval=rmspe, verbose_eval=True,evals_result=res)
    
    print("Validating")
    train_probs = gbm.predict(xgb.DMatrix(val_data[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe_score(np.exp(train_probs) - 1, val_data['Sales'].values)
    print('error', error)

    xgboost_train_end = time.time() - xgboost_start
    
    gbm_list.append(gbm)

# Predict first subsample 
print("Make predictions on the test set")
submission = pd.DataFrame([])
for s,gbm in enumerate(gbm_list):
    small_sample = range(s+1,len(store)+1,sub_range)
    org_test_sum_data = org_test_data[org_test_data.Store.isin(small_sample)]
    test = build_features(org_test_sum_data)
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    not_open_Id = test[test['Open']==0].index
    temp = pd.DataFrame({"Id": test["Id"], "Sales": np.expm1(test_probs)})
    temp.set_value(not_open_Id,"Sales",0.0)
    submission = pd.concat([submission,temp],axis=0)

submission['Id'] = pd.to_numeric(submission['Id'], downcast='integer')
submission.to_csv("my_submission_concat_10_2.csv", index=False)

# Strat second subsample training
gbm_list = []
sub_range = 5
for s in range(1,sub_range+1):
    small_sample = range(s,len(store)+1,sub_range)

    org_train_sum_data = org_train_data[org_train_data.Store.isin(small_sample)]

    val_start_date = org_train_sum_data.iloc[-1].Date - timedelta(weeks=4)
    mask = (org_train_sum_data['Date'] >= val_start_date) & (org_train_sum_data['Date'] <= org_train_sum_data.iloc[-1].Date)

    val_data = org_train_sum_data.loc[mask]
    train_data = org_train_sum_data.loc[~mask]
    train_data = build_features(train_data)
    val_data = build_features(val_data)
    
    xgboost_start = time.time()

    features = [x for x in train_data.columns if x not in ['Sales']]

    dtrain = xgb.DMatrix(train_data[features], np.log(train_data["Sales"] + 1))
    dvalid = xgb.DMatrix(val_data[features], np.log(val_data["Sales"] + 1))
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    params = {"objective": "reg:linear",
              "booster" : "gbtree",
              "eta": 0.01,
              "max_depth": 10,
              "subsample": 0.3,
              "colsample_bytree": 0.8,
              "min_child_weight":8,
              "reg_alpha":1e-04,
              "seed":3131,
              "silent": 1
              }
    
    num_trees = 30000
    
    res = {}
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=350, feval=rmspe, verbose_eval=True,evals_result=res)
    
    print("Validating")
    train_probs = gbm.predict(xgb.DMatrix(val_data[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe_score(np.exp(train_probs) - 1, val_data['Sales'].values)
    print('error', error)

    xgboost_train_end = time.time() - xgboost_start
    
    gbm_list.append(gbm)

# Predict second subsample 
print("Make predictions on the test set")
submission = pd.DataFrame([])
for s,gbm in enumerate(gbm_list):
    small_sample = range(s+1,len(store)+1,sub_range)
    org_test_sum_data = org_test_data[org_test_data.Store.isin(small_sample)]
    test = build_features(org_test_sum_data)
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    not_open_Id = test[test['Open']==0].index
    temp = pd.DataFrame({"Id": test["Id"], "Sales": np.expm1(test_probs)})
    temp.set_value(not_open_Id,"Sales",0.0)
    submission = pd.concat([submission,temp],axis=0)

# Combining two subsample results and average for final score
sub_10_2 = pd.read_csv('./my_submission_concat_10_2.csv')
sub_5_4 = pd.read_csv('./my_submission_concat_5_4.csv')

final = pd.merge(sub_10_2,sub_5_4,on='Id')
final['Sales'] = final['Sales_x']*0.5+final['Sales_y']*0.5
final.drop(['Sales_x','Sales_y'],axis=1,inplace=True)
display(final)
final.to_csv("my_submission_merge.csv", index=False)


