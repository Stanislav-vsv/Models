# coding: utf-8
import argparse
from time import gmtime, strftime
parser = argparse.ArgumentParser(description='MLM Environment Variables')
parser.add_argument('--login', default='VKOKHTEV')
parser.add_argument('--password', default='VKOKHTEV$')
parser.add_argument('--tns', default="""(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=dsacrmap)(PORT=1521)))(CONNECT_DATA=(SID=DMMLMPLT)(SERVER=DEDICATED)))""")
parser.add_argument('--job', default=strftime("%y%m%d%H%M%S9900002", gmtime()))
args = vars(parser.parse_known_args()[0])
#print("JOB:"+args["job"]+" Login:"+args["login"]+" Pwd:"+args["password"]+" TNS:"+args["tns"])

import pandas as pd
import numpy as np
pd.set_option("display.max_columns",40)
import warnings
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText
from sklearn import preprocessing
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
import time
from sklearn.metrics import average_precision_score
import cx_Oracle
import datetime

pd.set_option("display.max_columns",40)
warnings.filterwarnings('ignore')
np.random.seed(0)

###### Get connection
conn_bi = cx_Oracle.connect(args["login"],args["password"], args["tns"])
cursor_bi = conn_bi.cursor()

################### НЕФТЬ ###################
oil_str = "select * from MACRODATA.BRENT"
oil = pd.read_sql(oil_str, conn_bi)
#++++++++++++++++++++++++++++
# oil = pd.read_csv('Brent.csv',sep=';')
# oil.columns = ['VALUE_DAY','TRADE_OPEN']
# oil['TRADE_OPEN'] = oil['TRADE_OPEN'].apply(lambda x: x.replace(',','.')).astype(float)
# oil['VALUE_DAY'] = pd.to_datetime(oil['VALUE_DAY'],format='%d.%m.%Y')
#++++++++++++++++++++++++++++
oil.sort_values(by=['VALUE_DAY'],inplace=True)
oil.index = oil['VALUE_DAY']
oil = oil.resample("1d").interpolate("time")[['TRADE_OPEN']]
oil.columns=['value']
oil['DATE'] = oil.index
oil.name = 'oil'

################### ПРОЦЕНТНАЯ СТАВКА ###################
miacr_str = "select * from MACRODATA.MIACR"
miacr = pd.read_sql(miacr_str, conn_bi)
#++++++++++++++++++++++++++++
# miacr = pd.read_csv('MIACR.csv',sep=';')
# miacr.columns = ['VALUE_DAY','LAST_QUOTE_CLOSE']
# miacr['LAST_QUOTE_CLOSE'] = miacr['LAST_QUOTE_CLOSE'].apply(lambda x: x.replace(',','.'))
# miacr['VALUE_DAY'] = pd.to_datetime(miacr['VALUE_DAY'],format='%d.%m.%Y')
#++++++++++++++++++++++++++++
miacr.sort_values(by=['VALUE_DAY'],inplace=True)
lower_miacr = pd.DataFrame(miacr.iloc[-1,:].copy()).T
lower_miacr['VALUE_DAY'] += pd.Timedelta(30,unit='d')
upper_miacr = pd.DataFrame(miacr.iloc[0,:].copy()).T
upper_miacr['VALUE_DAY'] -= pd.Timedelta(30,unit='d')
miacr = pd.concat([upper_miacr,miacr,lower_miacr],axis=0)
miacr['LAST_QUOTE_CLOSE'] = miacr['LAST_QUOTE_CLOSE'].astype(float)
miacr.index = pd.to_datetime(miacr['VALUE_DAY'])
miacr = miacr.resample("1d").interpolate("time")[['LAST_QUOTE_CLOSE']]
miacr.columns=['value']
miacr['DATE'] = miacr.index
miacr.name = 'miacr'

################### ВВП ###################
gdp_str = "select * from MACRODATA.GDP"
gdp = pd.read_sql(gdp_str, conn_bi)

lower_gdp = pd.DataFrame(gdp.iloc[-1,:].copy()).T
lower_gdp['VALUE_DAY'] -= pd.Timedelta(95,unit='d')
upper_gdp = pd.DataFrame(gdp.iloc[0,:].copy()).T
upper_gdp['VALUE_DAY'] += pd.Timedelta(365,unit='d')

gdp = pd.concat([upper_gdp,gdp,lower_gdp],axis=0)
gdp.index = pd.to_datetime(gdp['VALUE_DAY'])
gdp['value'] = gdp['GDP_USD_BN_2011'].astype(int)
gdp.drop(['VALUE_DAY','GDP_USD_BN_2011','GDP_USD_BN_2016'],axis=1,inplace=True)
gdp = gdp.resample("1d").interpolate("time")

gdp['DATE'] = gdp.index
gdp.name = 'gdp'


################### ИНФЛЯЦИЯ, БЕЗРАБОТИЦА ###################
inf_unemp_str = "select * from MACRODATA.INFLAT_UNEMPLOY"
inf_unemp = pd.read_sql(inf_unemp_str, conn_bi)

lower_inf_unemp = pd.DataFrame(inf_unemp.iloc[-1,:].copy()).T
lower_inf_unemp['VALUE_DAY'] -= pd.Timedelta(60,unit='d')
upper_inf_unemp = pd.DataFrame(inf_unemp.iloc[0,:].copy()).T
upper_inf_unemp['VALUE_DAY'] += pd.Timedelta(180,unit='d')

inf_unemp = pd.concat([upper_inf_unemp,inf_unemp,lower_inf_unemp],axis=0)
inf_unemp[['INFLAT','UNEMPLOY']] = inf_unemp[['INFLAT','UNEMPLOY']].astype(float)
inf_unemp.index = pd.to_datetime(inf_unemp['VALUE_DAY'])
del inf_unemp['VALUE_DAY']
inf_unemp = inf_unemp.resample("1d").interpolate("time")
inf_unemp['DATE'] = inf_unemp.index

################### ИНФЛЯЦИЯ ###################
inflation = inf_unemp[['DATE','INFLAT']].copy()
inflation.columns = ['DATE','value']
inflation.name = 'inflation'

################### БЕЗРАБОТИЦА ###################
unemp = inf_unemp[['DATE','INFLAT']].copy()
unemp.columns = ['DATE','value']
unemp.name = 'unemp'


################### КУРСЫ ВАЛЮТ ###################
exch_str = "select * from GUARANT.EXCHANGERATE_STAT"
exch = pd.read_sql(exch_str, conn_bi)
exch.drop(['XK','XRATETYPE_UK','CURRENCY_FROM_UK','CURRENCY_TO_UK',\
          'DWSCMIX','JOB_INSERT','AS_OF_DAY','DQ_MISTAKECOST_UK',\
          'DQ_CONTROL_UK_LIST','JOB_UPDATE','DELETED_FLAG','CURRENCY_FROM_CCODE'],axis=1,inplace=True)
exch.sort_values(by=['VALUE_DAY'],inplace=True)
exch.index = exch['VALUE_DAY']
exch.columns = ['DATE','value','code']

euro, dollar = exch[exch['code'] == 'EUR'], exch[exch['code'] == 'USD']
euro.name, dollar.name = 'euro', 'dollar'
del euro['code'], dollar['code'], exch


# gdp = pd.read_csv('gdp1.csv',encoding='utf-8',sep=';',index_col=0)
# inflation = pd.read_csv('inflation.csv',encoding='utf-8',sep=';',index_col=0)
# unemp = pd.read_csv('unemp.csv',encoding='utf-8',sep=';',index_col=0)
# euro = pd.read_csv('euro.csv',encoding='utf-8',sep=';',index_col=0)
# dollar = pd.read_csv('dollar.csv',encoding='utf-8',sep=';',index_col=0)

for df in [oil, miacr, gdp, inflation, unemp, euro, dollar]:
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.index = pd.to_datetime(df.index)
    df['num'] = np.arange(df.shape[0])

data = pd.read_sql("select * from GUARANT.DEALGUARANTEE", conn_bi)
# data = pd.read_csv('data.csv',sep=';',index_col=0)


CURRENT_DAY = pd.to_datetime(data['VALUE_DAY'][0])
for col in ['START_DATE','END_PLAN_DATE','END_FACT_DATE']:
    data[col] = pd.to_datetime((data[col].astype(str)).apply(lambda x: x.replace('5999','2030')))
    
data = data[(data['DEAL_CURRENCY'] == 'RUR') | (data['DEAL_CURRENCY'] == 'USD') | (data['DEAL_CURRENCY'] == 'EUR')]
data.sort_values(by=['START_DATE'],inplace=True)
# Оставляем данные в заданных временных границах
data = data[(data['START_DATE'] >= pd.to_datetime('01.01.2012')) & (data['END_PLAN_DATE'] < (CURRENT_DAY + pd.Timedelta(30,unit='D')))] ## тут текущий день (есть в таблице) + 30 дней
######## 2012
data.drop(['VALUE_DAY','DEAL_FRONT_REF','MODULE_FRONT_CCODE',
           'DEAL_BACK_REF','MODULE_BACK_CCODE',
           'DEALCLASS_CCODE','DEALKIND_CCODE','BENEFICIAR_PIN',
           'BRANCH_EQ_CCODE','PROFITCENTER_ENG_CCODE',
           'LOANPURPOSE_CCODE','ADVREPAYRIGHT_CCODE','PRODUCT_CCODE',
           'DEAL_CUR_AMT','DELAY_FLAG'],1,inplace=True)

data['BENEFICIAR_NAME'].fillna('Не определено',inplace=True)
data[['COMISSION_RATE','USED_AMOUNT']] = data[['COMISSION_RATE','USED_AMOUNT']].fillna(0)
data['USED_CURRENCY'].fillna('RUR',inplace=True)
data.drop_duplicates(subset=['DEAL_UK'],keep='last',inplace=True)
data['DURATION'] = (data['END_PLAN_DATE'] - data['START_DATE']).apply(lambda x: x.days)

data.index = np.arange(data.shape[0])

def sum_to_roubles(price,curr_type,time):
    if curr_type == 'USD':
        return (price * dollar[dollar['DATE'] == time]['value']).values[0]
    elif curr_type == 'EUR':
        return (price * euro[euro['DATE'] == time]['value']).values[0]
    else:
        return price


# В пустой лист запишем сумму в рублях для каждой сделки и добавим лист как признак в таблицу
sum_arr = []
used_arr = []
for i in (data.index):
    row = data.iloc[i,:]
    day = min(row['END_PLAN_DATE']-pd.Timedelta(30,unit='d'),CURRENT_DAY-pd.Timedelta(3,unit='d')).replace(hour=0, minute=0, second=0)
    sum_arr.append(sum_to_roubles(row['DEAL_START_AMOUNT'],row['DEAL_CURRENCY'],day))
    used_arr.append(sum_to_roubles(row['USED_AMOUNT'],row['USED_CURRENCY'],day))
data['DEAL_AMOUNT_RUB'] = sum_arr
data['USED_AMOUNT'] = used_arr
data.drop(['DEAL_START_AMOUNT','USED_CURRENCY'],axis=1,inplace=True)
del sum_arr

# Простейшие замены с неплохим качеством
data['PRINCIPAL_NAME'] = data['PRINCIPAL_NAME'].apply(lambda x: x.lower().replace('федеральное государственное унитарное предприятие','фгуп').replace('открытое акционерное общество','оао').replace('публичное акционерное общество','пао').replace('акционерное общество','ао').replace('закрытое акционерное общество','зао').replace('общество с ограниченной ответственностью','ооо').replace('общество с ограниченной ответсвенностью','ооо').replace('закрытое ао','зао').replace('"','').replace('.','').replace(',',''))
data['BENEFICIAR_NAME'] = data['BENEFICIAR_NAME'].apply(lambda x: x.lower().replace('федеральное государственное унитарное предприятие','фгуп').replace('открытое акционерное общество','оао').replace('публичное акционерное общество','пао').replace('акционерное общество','ао').replace('закрытое акционерное общество','зао').replace('общество с ограниченной ответственностью','ооо').replace('общество с ограниченной ответсвенностью','ооо').replace('закрытое ао','зао').replace('"','').replace('.','').replace(',',''))
# Если первая строка до пробела в list - возвращаем её, иначе 'неизвестно'
def org_form(x):
    if x.split(' ')[0] in ['фгуп','оао','ооо','ао','зао','пао']:
        return x.split(' ')[0]
    else:
        return 'неизвестно'
data['PRINCIPAL_ORG'] = data['PRINCIPAL_NAME'].apply(org_form)
data['BENEFICIAR_ORG'] = data['BENEFICIAR_NAME'].apply(org_form)

def add_timestamps(data,N_parts):
    result = data.copy()
    t = pd.Series([CURRENT_DAY.replace(hour=0, minute=0, second=0)-pd.Timedelta(2,unit='d')]*data.shape[0])
    last_t = pd.DataFrame([data['END_PLAN_DATE'],t]).T.min(axis=1)
    for i in range(N_parts):
        result['timestamp_'+str(i)] = result['START_DATE'].values + pd.to_timedelta((i*(last_t-result['START_DATE'])/(N_parts-1)).apply(lambda x: x.days), unit='d')
    return result

def prediction(values,current_value):
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(np.array([1,45,90]).reshape(-1, 1),np.array(values))#[20.02,17.51,15.63]
    return np.array([m.coef_[0]*(x-1)+current_value for x in range(1,32)])

def stress_values(df, fit_values, row,is_bau=False):
    stress_date = max(row['START_DATE'],row['END_PLAN_DATE'] - pd.Timedelta(30,unit='d'))
    bau_val = df.loc[:stress_date,'value']
    current_value = bau_val.values[-1]
    cols = [x for x in row.keys() if 'timestamp_' in x]
    if row['DURATION'] < 6:
        return len(cols)*[current_value]
    indexes = list(df[df['DATE'].isin(row[cols])]['num'])
    #print(indexes)
    if is_bau:
        bau_val = df.loc[:,'value']
        return bau_val.values[indexes]
    else:
        pred_stress = prediction(fit_values, current_value)
        return np.hstack((bau_val.values,pred_stress))[indexes]

def create_overall(df,aggregate_method,N_parts):
    add = '_'
    for macro in ['oil','miacr','inflation','unemp','gdp','dollar','euro']:
        if aggregate_method == 'mean':
            df[macro] = df[[macro+add+str(i) for i in range(N_parts)]].T.mean()
        if aggregate_method == 'median':
            df[macro] = df[[macro+add+str(i) for i in range(N_parts)]].T.median()
        if aggregate_method == 'max':
            if macro in ['gdp','miacr','oil']:
                df[macro] = df[[macro+add+str(i) for i in range(N_parts)]].T.min()
            else:
                df[macro] = df[[macro+add+str(i) for i in range(N_parts)]].T.max()
        df.drop([macro+add+str(i) for i in range(N_parts)],1,inplace=True)
    return df


macro_dfs = [[oil, [20.02,17.51,15.63]],\
    [gdp, list(np.array([100-8.77,100-8.68,100-9.5])*150)],\
    [inflation, [13.22,15.56,17.68]],\
    [miacr, [15.96,18.12,19.69]],\
    [unemp, [9,10.06,11.01]],\
    [euro, [94.18,105.37,116.16]],\
    [dollar, [94.18,105.37,116.16]]]

macro = []
macro_stress = []   

N_parts = 5

data = add_timestamps(data,N_parts)
data_stress = add_timestamps(data,N_parts)

for index, row in (data_stress.iterrows()):
    vector = np.hstack([stress_values(df, fit_values, row, True) for df, fit_values in macro_dfs])
    vector_stress = np.hstack([stress_values(df, fit_values, row, False) for df, fit_values in macro_dfs])
    macro.append(vector)
    macro_stress.append(vector_stress)
    
macro_columns = ['oil','gdp','inflation','miacr','unemp','euro','dollar']
macro_cols = [x+'_'+str(i) for x in macro_columns for i in range(N_parts)]
macro_stress_df = create_overall(pd.DataFrame(macro_stress,columns=macro_cols), 'max', N_parts)
macro_df = create_overall(pd.DataFrame(macro,columns=macro_cols), 'max', N_parts)

data = pd.concat([data.loc[:,:'BENEFICIAR_ORG'],macro_df],axis=1)
data_stress = pd.concat([data_stress.loc[:,:'BENEFICIAR_ORG'],macro_stress_df],axis=1)


pd.Series('\t'.join(data['PRINCIPAL_NAME'] +' '+ data['BENEFICIAR_NAME'])).to_csv('text.txt',sep='\t',index=None)
string = LineSentence('text.txt')
fasttext = FastText(size=50,sg=0,word_ngrams=2,iter=10,min_n=2,max_n=10)
fasttext.build_vocab(string)
fasttext.train(string, total_examples=fasttext.corpus_count, epochs=fasttext.iter)

def to_sent_emb(string):
    return np.sum([fasttext[x] for x in string.split(' ') if len(x)>1],axis=0)

emb_df = pd.DataFrame(np.vstack((data['PRINCIPAL_NAME'] +' '+ data['BENEFICIAR_NAME']).apply(to_sent_emb)))
emb_df.columns = ['emb_'+str(w) for w in emb_df.columns]

data = pd.concat([data,emb_df],axis=1)
data.drop(['PRINCIPAL_PIN','PRINCIPAL_NAME','BENEFICIAR_NAME','USED_FLAG','USED_AMOUNT'],axis=1,inplace=True)

target = (data_stress['USED_FLAG'] == 'Y').astype(int)

data_stress = pd.concat([data_stress, emb_df],axis=1)
data_stress.drop(['PRINCIPAL_PIN','PRINCIPAL_NAME','BENEFICIAR_NAME','USED_FLAG'],axis=1,inplace=True)

cat_columns = ['DEAL_CURRENCY','PRODUCT_NAME','DEALCLASS_NAME','DEALKIND_NAME',\
            'SALESPLACE_NAME','PROFITCENTER_NAME','LOANPURPOSE_NAME','ADVREPAYRIGHT_NAME',\
           'PRINCIPAL_ORG', 'BENEFICIAR_ORG']

def counts(column,Y,C):
    counts = dict()
    global_mean = (Y == 1).sum()/Y.shape[0]
    for elem in np.unique(column):
        #print(elem)
        counts[elem] = (max(0,(((column == elem) & (Y == 1)).sum())) + global_mean*C)/((column == elem).sum()+C)
    return counts

def create_cat_features(X,Y,cat_columns,catboost=True):
    X1 = X.copy()
    if catboost == True:
        for column in cat_columns:
            le1 = preprocessing.LabelEncoder()
            X1[column] = le1.fit_transform(X1[column])
        
    if catboost == False:
        for column in cat_columns:
            X1 = X1.replace({column:counts(X1[column],Y,0.5)})
    return X1,Y

data_stress, target_stress = create_cat_features(data_stress, target, cat_columns, catboost=True)
data, target = create_cat_features(data, target, cat_columns, catboost=True)

data_train, data_test = data[data['END_PLAN_DATE'] < CURRENT_DAY], data[data['END_PLAN_DATE'] >= CURRENT_DAY]
target_train = target.loc[data_train.index]

data_stress_train, data_stress_test = data_stress[data_stress['END_PLAN_DATE'] < CURRENT_DAY],\
                                            data_stress[data_stress['END_PLAN_DATE'] >= CURRENT_DAY]
target_stress_train = target.loc[data_stress_train.index]

def metric(price, y_true, y_pred):
    true_val = sum(np.multiply(y_true,price))
    if true_val == 0:
        return 1
    else:
        return float(sum(np.multiply(y_pred,price)) / true_val)

def cross_val(X, Y, estimator_type, model):
    K = 3
    kf = StratifiedKFold(n_splits=K)
    test_answers = []
    pr, metric = 0, 0
    for train, test in kf.split(X,Y):
        X_train, X_test = X.iloc[train,:], X.iloc[test,:]
        y_train, y_test = Y.iloc[train], Y.iloc[test]
        if estimator_type == 'catboost':
            model.fit(X_train,y_train,cat_features=[i for i, c in enumerate(X_train.columns) if c in cat_columns])
            y_pred = model.predict_proba(X_test)[:,1]
            test_answers.append([np.array(X_test['DEAL_AMOUNT_RUB']),np.array(y_test),y_pred, model.get_params()])
    return test_answers

def find_threshold(price, y_true, y_pred):
    metric_arr = np.array([metric(price, y_true, (y_pred > (t/400)).astype(int)) for t in range(200)])
    ind = np.where(metric_arr > 1)
    if ind[0].shape[0] == 0:
        return 0
    return ind[0][int(np.argsort(metric_arr[ind])[0])] / 400
    
def grid_search(estimator_type, param_grid, X, Y):
    overall_search = []
    for params in (param_grid):
        if estimator_type == 'catboost':
            model = CatBoostClassifier(**params)
            test_answers = cross_val(X, Y, estimator_type, model)
            overall_search.append(test_answers)
    best_pr = 0
    best_metric = 1000000
    best_estimator = None
    time.sleep(1)
    for obs in (overall_search):
        obs_metric, obs_pr, t_mean = 0, 0, []
        for price, y_test, y_pred, param in obs:
            t = find_threshold(price, y_test, y_pred)
            t_mean.append(t)
            y = (y_pred > t).astype(int)
            obs_metric += metric(price, y_test, y) / 3
            obs_pr += average_precision_score(y_test, y) / 3
        if obs_metric < best_metric:
            best_pr, best_metric, best_estimator, best_t = obs_pr, obs_metric, param, t_mean
        else:
            continue
    return [best_pr, best_metric, best_estimator, t_mean]


# param_grid = ParameterGrid({'learning_rate': [0.0005,0.001,0.005,0.01,0.03,0.07,0.1], 'iterations': [500,300,100], 'depth': [5,6,7],\
#               'thread_count':[8], 'random_seed':[0],'logging_level':['Silent']})
# print(' BAU')
# bau_params = grid_search('catboost', param_grid, data_train.loc[:,'DEAL_CURRENCY':], target_train)
# print('\n','Stress')
# stress_params = grid_search('catboost', param_grid, data_stress_train.loc[:,'DEAL_CURRENCY':], target_stress_train)

bau_params = dict({'depth': 5,
  'iterations': 300,
  'learning_rate': 0.01,
  'loss_function': 'Logloss',
  'random_seed': 0})
#bau_threshold 0.0075
## pr-roc 0.662
## metric 1.11627

stress_params = dict({'depth': 6,
  'iterations': 300,
  'learning_rate': 0.0005,
  'loss_function': 'Logloss',
  'random_seed': 0})
#stress_threshold 0.0
## pr-roc 0.640
## metric 1.03884

def predict(model_type, X_train, y_train, X_test, estimator_type):
    if model_type == 'bau':
        threshold = 0.0005
        params = bau_params.copy()
    if model_type == 'stress':
        threshold = 0.001
        params = stress_params.copy()
    
    if estimator_type == 'catboost':
        model = CatBoostClassifier(**params)
        model.fit(X_train,y_train,cat_features=[i for i, c in enumerate(X_train.columns) if c in cat_columns])
    y_pred = model.predict_proba(X_test)[:,1]
    model_result = sum(np.multiply((y_pred > threshold).astype(int),X_test['DEAL_AMOUNT_RUB']))
    
    return model_result

bau_result = predict('bau', data_train.loc[:,'DEAL_CURRENCY':], target_train, data_test.loc[:,'DEAL_CURRENCY':], 'catboost')

stress_result = predict('stress', data_stress_train.loc[:,'DEAL_CURRENCY':], target_stress_train, data_stress_test.loc[:,'DEAL_CURRENCY':], 'catboost')

pkl_result = 0.1 * sum(data_stress_test['DEAL_AMOUNT_RUB'])

guarant_result = pd.DataFrame([[CURRENT_DAY,bau_result,stress_result,pkl_result]],columns=['VALUE_DAY','BAU','STRESS','PKL'])

###### Write Results to OUTPUT table

cursor_bi.prepare("""insert into GUARANT.GUARANT_OUT (VALUE_DAY, BAU, STRESS, PKL, JOB) 
                  values (:1, :2, :3, :4, """+args["job"]+")")
cursor_bi.executemany(None, guarant_result.values.tolist())
conn_bi.commit()

conn_bi.close()