
###############################   model_guarantees.py  ############################### 

# coding: utf-8

import pandas as pd   # Пакет pandas в Python 3 используется для управления и анализа данных (в том числе помеченных и реляционных данных).   pd пространство имен 
import numpy as np    # NumPy — это расширение языка Python, добавляющее поддержку больших многомерных массивов и матриц, вместе с большой библиотекой высокоуровневых математических функций для операций с этими массивами.
pd.set_option("display.max_columns",40)
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
np.random.seed(0)

def macro_df(macro,t):
    date = pd.date_range(t, t+pd.Timedelta(30,unit='d'), freq='D') ## current day
    
    name = macro.name
    macro = macro[(macro.index < t)] ## current day
    pred = pd.DataFrame(np.array([prediction([1,1,1], macro['value'].values[-1]),pd.to_datetime(date)]).T,columns=['value','DATE'])
    macro['DATE'] = pd.to_datetime(macro['DATE'])#.apply(lambda x: x.to_timestamp())
    macro = pd.concat([macro,pred],axis=0)
    macro.index = macro['DATE']
    macro.name = name
    return macro

def prediction(values,current_value):
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(np.array([1,45,90]).reshape(-1, 1),np.array(values))#[20.02,17.51,15.63]
    k = m.coef_[0]
    b = current_value - k
    return np.array([k*x+b for x in range(1,32)])

# подразумевается, что файлы для макрофакторов будут в лежать в некоторых таблицах

# НЕФТЬ https://www.finam.ru/profile/tovary/brent/export/?market=24&em=19473&code=BZ&apply=0&df=1&mf=0&yf=2010&from=01.01.2010&dt=23&mt=9&yt=2017&to=23.10.2017&p=8&f=BZ_100101_171023&e=.csv&cn=BZ&dtf=4&tmf=1&MSOR=1&mstime=on&mstimever=1&sep=3&sep2=1&datf=5&at=1&fsp=1
oil = pd.read_csv('./data/brent.csv', sep = ';',index_col=0)
oil.drop(['<TIME>','<CLOSE>','<HIGH>','<LOW>','<VOL>'],1,inplace=True)
oil.index = pd.to_datetime(oil.index + 20000000, format='%Y%m%d')
oil = oil.resample("1d").interpolate("time")[['<OPEN>']]
oil.columns=['value']
oil['DATE'] = oil.index
oil.name = 'oil'
oil = macro_df(oil,pd.to_datetime('01.09.2017',format='%d.%m.%Y'))

# ВВП http://www.gks.ru/free_doc/new_site/vvp/kv/tab6.htm
gdp_q = pd.read_excel('./data/tab6.xlsx').iloc[4].values
gdp_q = pd.DataFrame(gdp_q,columns=['value'],index=np.array([[str(i)]*4 for i in range(2011,2018)]).reshape(1,28)[0])
gdp_q.index = pd.to_datetime(gdp_q.index + np.array(['1','4','7','10']*7), format='%Y%m')
start_date = gdp_q.index.min() - pd.DateOffset(day=1)
end_date = gdp_q.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='D')
gdp = gdp_q.reindex(dates, method='ffill')
gdp['DATE'] = gdp.index
del gdp_q
gdp.name = 'gdp'
gdp = macro_df(gdp,pd.to_datetime('01.09.2017',format='%d.%m.%Y'))

# ИНФЛЯЦИЯ https://www.statbureau.org/ru/russia/inflation-tables
inflation = pd.read_csv('./data/inflation.csv', sep = ',').iloc[19:,:13]
inflation.index = np.arange(inflation.shape[0])
inf_values = []
for year in range(inflation.shape[0]):
    for elem in inflation.iloc[year,1:].values:
        inf_values.append([str(inflation['Год'][year]),elem])
inflation = pd.DataFrame(inf_values,index=np.array([str(i) for i in range(1,13)] * 8),columns=['year','value'])
inflation.index = pd.to_datetime(inflation['year']+inflation.index, format='%Y%m')
del inflation['year']
del inf_values
start_date = inflation.index.min() - pd.DateOffset(day=1)
end_date = inflation.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='D')
inflation = inflation.reindex(dates, method='ffill')
inflation['DATE'] = inflation.index
inflation.name = 'inflation'
inflation = macro_df(inflation,pd.to_datetime('01.09.2017',format='%d.%m.%Y'))

# ПРОЦЕНТНАЯ СТАВКА http://www.cbr.ru/hd_base/mkr/mkr_base/
miacr = pd.read_csv('./data/MIACR.csv',sep=';',index_col=0)
miacr = miacr.iloc[::-1]
miacr.columns=['value']
miacr.index = pd.to_datetime(miacr.index, format='%d.%m.%Y')
miacr = miacr.resample("1d").interpolate("time")
miacr['DATE'] = miacr.index
miacr.name = 'miacr'
miacr = macro_df(miacr,pd.to_datetime('01.09.2017',format='%d.%m.%Y'))

# БЕЗРАБОТИЦА https://www.finam.ru/analysis/macroevent/?dind=0&dpsd=139530&fso=date+desc&str=1&ind=545&stdate=19.02.2010&endate=18.10.2017&sema=0&seman=5&timeStep=2
unemp = pd.read_csv('./data/unemp.csv',sep=';')
unemp = pd.DataFrame(unemp.iloc[::-1])
unemp.drop(['Дата','Период','Пред'],1,inplace=True)
unemp.columns = ['value']
unemp.index = pd.to_datetime((pd.Series(np.array([str(i) for i in range(1,13)] * 8)) + (np.array([[str(i)]*12 for i in range(2010,2018)]).reshape(1,96)[0]))[:93], format='%m%Y')
start_date = unemp.index.min() - pd.DateOffset(day=1)
end_date = unemp.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='D')
unemp = unemp.reindex(dates, method='ffill')
unemp['DATE'] = unemp.index
unemp.name = 'unemp'
unemp = macro_df(unemp,pd.to_datetime('01.09.2017',format='%d.%m.%Y'))

#Валюта K/OSTAT/S_V.XLS
exch = pd.read_excel('./data/S_V.XLS')[['DATA','KURS_840','KURS_978']]
exch.index = exch['DATA']
exch = exch.resample("1d").interpolate("time")
euro = pd.DataFrame(exch['KURS_978'])
euro.columns = ['value']
euro['DATE'] = euro.index
euro.name = 'euro'
euro = macro_df(euro,pd.to_datetime('01.09.2017',format='%d.%m.%Y'))

dollar = pd.DataFrame(exch['KURS_840'])
dollar.columns = ['value']
dollar['DATE'] = dollar.index
dollar.name = 'dollar'
dollar = macro_df(dollar,pd.to_datetime('01.09.2017',format='%d.%m.%Y'))
del exch


data = pd.read_excel('./data/Гарантии_new.xlsx',header=[0]) ## Тут чтение из БД
data.drop(0, inplace=True)

# Удаляем ненужные поля
data.drop(['UK сделки на DWH','Номер сделки в Фронт-системе','Символьный код фронт системы','Номер сделки в Бэк-системе','Символьный код бэк системы','Пин принципала','Ставка комиссии','Символьный код разновидности сделки','Символьный код класса сделки','Пин бенефициара','Символьный код филиала','Символьный код продукта','Наименование продукта','Символьный код цели','Символьный код ПЦ англ.','Символьный код ПДП','Наименивание ПЦ'],1,inplace=True)

# Редкие валюты не рассматриваем
data = data[(data['Валюта сделки'] == 'RUR') | (data['Валюта сделки'] == 'USD') | (data['Валюта сделки'] == 'EUR')]

# Переводим поэлементно - были ошибки
for i in ['Дата начала','Плановая дата завершения','Фактическая дата завершения']:
    for j in data.index:
        try:
            data[i][j] = pd.to_datetime(data[i][j])
        except:
            data[i][j] = pd.to_datetime('01.01.2030')
            
# Оставляем данные в заданных временных границах
data = data[data['Дата начала'] >= pd.to_datetime('01.01.2011')]#[data['Плановая дата завершения'] < pd.to_datetime('01.09.2017',format='%d.%m.%Y')] ## тут текущий день (есть в таблице) + 30 дней
# Заполняем пропуски
data['Наименование бенефициара'] = data['Наименование бенефициара'].fillna('Неизвестно')
# Получаем столбец длительностей в днях
data['Длительность'] = (data['Плановая дата завершения'] - data['Дата начала']).apply(lambda x: x.days)

data.index = np.arange(data.shape[0])

def sum_to_roubles(price,curr_type,time):
    if curr_type == 'USD':
        return (price * dollar[dollar['DATE'] == time]['value']).values[0]
    elif curr_type == 'EUR':
        return (price * euro[euro['DATE'] == time]['value']).values[0]
    else:
        return price
    
def sum_to_roubles(price,curr_type,time):
    if curr_type == 'USD':
        return (price * dollar[dollar['DATE'] == time]['value']).values[0]
    elif curr_type == 'EUR':
        return (price * euro[euro['DATE'] == time]['value']).values[0]
    else:
        return price
# В пустой лист запишем сумму в рублях для каждой сделки и добавим лист как признак в таблицу
sum_arr = []
for i in data.index:
    row = data.iloc[i,:]
    current_day = min(row['Плановая дата завершения']-pd.Timedelta(30,unit='d'),pd.to_datetime('01.09.2017',format='%d.%m.%Y'))
    sum_arr.append(sum_to_roubles(row['Сумма сделки'],row['Валюта сделки'],current_day))
data['Сумма в рублях'] = sum_arr
del data['Сумма сделки']
del sum_arr

# Простейшие замены с неплохим качеством
data['Наименование принципала'] = data['Наименование принципала'].apply(lambda x: x.lower().replace('федеральное государственное унитарное предприятие','фгуп').replace('открытое акционерное общество','оао')    .replace('публичное акционерное общество','пао').replace('акционерное общество','ао').replace('закрытое акционерное общество','зао').replace('общество с ограниченной ответственностью','ооо').replace('общество с ограниченной ответсвенностью','ооо').replace('закрытое ао','зао').replace('"','').replace('.','').replace(',',''))
# Если первая строка до пробела в list - возвращаем её, иначе 'неизвестно'
def org_form(x):
    if x.split(' ')[0] in ['фгуп','оао','ооо','ао','зао','пао']:
        return x.split(' ')[0]
    else:
        return 'неизвестно'
data['Оргформа принципала'] = data['Наименование принципала'].apply(org_form)
# Тексты ничего не дали в экспериментах, поэтому удаляем
data.drop(['Наименование принципала','Наименование бенефициара','Наименование цели','Величина раскрытия','Валюта раскрытия'],1,inplace=True)
data = data.fillna('0')

def add_timestamps(data,N_parts,t):
    result = data.copy()
    for i in range(N_parts):
        result['timestamp_'+str(i)] = result['Дата начала'].values + pd.to_timedelta((i*(result['Плановая дата завершения']-result['Дата начала'])/(N_parts-1)).apply(lambda x: x.days), unit='d')
        # вместо Плановой даты завершения должна быть текущая дата 
    return result

def add_macro(df,N_parts,macro_params):
    macro_params = [macro_params[0]]+macro_params
    for macro in macro_params:
        for i in range(N_parts):
            df = pd.merge(df, macro, how='left',left_on='timestamp_'+str(i),right_on='DATE')
            df.rename(columns={'value':macro.name+'_'+str(i)}, inplace=True)
            del df['DATE']
            
    df.drop(['timestamp_'+str(i) for i in range(N_parts)],1,inplace=True)
    return df

def create_overall(df,aggregate_method,N_parts,stress):
    if stress:
        add = '_stress_'
    else:
        add = '_'
    for macro in ['oil','miacr','inflation','unemp','gdp','dollar','euro']:
        if aggregate_method == 'mean':
            df[macro+'_mean'] = df[[macro+add+str(i) for i in range(N_parts)]].T.mean()
        if aggregate_method == 'median':
            df[macro+'_median'] = df[[macro+add+str(i) for i in range(N_parts)]].T.median()
        if aggregate_method == 'max':
            if macro in ['gdp','miacr','oil']:
                df[macro+'_max'] = df[[macro+add+str(i) for i in range(N_parts)]].T.min()
            else:
                df[macro+'_max'] = df[[macro+add+str(i) for i in range(N_parts)]].T.max()
        df.drop([macro+add+str(i) for i in range(N_parts)],1,inplace=True)
    return df

def counts(column,Y,C):
    counts = dict()
    global_mean = (Y == 1).sum()/Y.shape[0]
    for elem in np.unique(column):
        #print(elem)
        counts[elem] = (max(0,(((column == elem) & (Y == 1)).sum())) + global_mean*C)/((column == elem).sum()+C)
    return counts

def create_cat_features(X,Y,cat_columns,catboost=False):
    X1 = X.copy()
    if catboost == True:
        for column in cat_columns:
            le1 = preprocessing.LabelEncoder()
            X1[column] = le1.fit_transform(X1[column])
        
    if catboost == False:
        for column in cat_columns:
            X1 = X1.replace({column:counts(X1[column],Y,0.5)})
    return X1,Y

def predict_macro_factor(df,t,ref_array,stress=False):
    dates = pd.date_range(t, t+pd.Timedelta(30,unit='d'), freq='D')
    dataset = df.copy()
    dataset_values = dataset[(dataset.index < t)]#
    pred = pd.DataFrame(np.array([prediction(ref_array, dataset_values['value'].values[-1]),pd.to_datetime(dates)]).T,columns=['value','DATE'])
    dataset = pd.concat([dataset_values,pred],axis=0)
    dataset.index = dataset['DATE']
    if stress:
        add = '_stress'
    else:
        add = ''
    dataset.name = df.name+add
    return dataset

def create_stress_by_date(t):
    
    oil_stress = predict_macro_factor(oil, t, [20.02,17.51,15.63],stress=True)
    gdp_stress = predict_macro_factor(gdp, t, np.array([100-8.77,100-8.68,100-9.5])*150,stress=True)
    inflation_stress = predict_macro_factor(inflation, t, [13.22,15.56,17.68],stress=True)
    miacr_stress = predict_macro_factor(miacr, t, [15.96,18.12,19.69],stress=True)
    unemp_stress = predict_macro_factor(unemp, t, [9,10.06,11.01],stress=True)
    euro_stress = predict_macro_factor(euro, t, [94.18,105.37,116.16],stress=True)
    dollar_stress = predict_macro_factor(dollar, t, [94.18,105.37,116.16],stress=True)
                                      
    return [oil_stress,miacr_stress,inflation_stress,unemp_stress,gdp_stress,dollar_stress,euro_stress]

def split(df,target,t):
    dataset_train = df[df['Плановая дата завершения'] < t]
    dataset_test = df[df['Плановая дата завершения'] >= t]
    dataset_train.drop(['Дата начала','Плановая дата завершения','Фактическая дата завершения'],axis=1,inplace=True)
    dataset_test.drop(['Дата начала','Плановая дата завершения','Фактическая дата завершения'],axis=1,inplace=True)
    target_train = target.loc[dataset_train.index]
    target_test = target.loc[dataset_test.index]
    return dataset_train, dataset_test, target_train, target_test

def dataset(df,aggregate_method,N_parts,t,scaler=None,stress=False):
    dataset = df.copy()
    dataset = dataset[dataset['Плановая дата завершения'] <= (t+pd.Timedelta(30,unit='d'))]
    dataset = add_timestamps(dataset,N_parts,t)
    
    if stress:
        macro_params = create_stress_by_date(t) ## текущая дата
    else:
        macro_params = [oil,miacr,inflation,unemp,gdp,dollar,euro]
        
    dataset = add_macro(dataset,N_parts,macro_params)
    #return dataset, 0
    dataset = create_overall(dataset,aggregate_method,N_parts,stress=stress)
    
    target = (dataset['Признак раскрытия'] == 'Y').astype(int)
    dataset.drop(['Признак раскрытия','Признак просрочки'],axis=1,inplace=True)
    cat_columns = ['Оргформа принципала','Валюта сделки','Наименование класса сделки','Наименование разновидности сделки','Наименование филиала','Наименование ПДП']

    dataset, target = create_cat_features(dataset,target,cat_columns,catboost=False)
    
    cat_columns = ['Дата начала','Плановая дата завершения','Фактическая дата завершения'] + cat_columns
    num_columns = [item for item in list(dataset.columns) if item not in cat_columns]
    
    if scaler == 'Standart':
        from sklearn import preprocessing
        sc = preprocessing.StandardScaler()
        print(sc)
    if scaler == 'MinMax':
        from sklearn import preprocessing
        sc = preprocessing.MinMaxScaler()
    if scaler == None:
        return split(dataset,target,t)
    
    dataset = pd.DataFrame(np.hstack([dataset[cat_columns],sc.fit_transform(dataset[num_columns])]),columns=cat_columns+num_columns)
    return split(dataset,target,t)

t = pd.to_datetime('01.09.2017',format='%d.%m.%Y') 

X_bau_train, X_bau_test, Y_bau_train, Y_bau_test = dataset(data,'max',10,t,scaler=None,stress=False)

X_stress_train, X_stress_test, Y_stress_train, Y_stress_test = dataset(data,'max',10,t,scaler=None,stress=True)

price_vector = X_stress_test['Сумма в рублях']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=600, criterion='gini', max_depth=8, max_features='sqrt', n_jobs=-1)
clf_stress = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=7, max_features='sqrt', n_jobs=-1)
clf.fit(X_bau_train,Y_bau_train)
clf_stress.fit(X_stress_train,Y_stress_train)
Y_bau_pred = (clf.predict(X_bau_test) > 0.091091).astype(int)
Y_stress_pred = (clf_stress.predict(X_stress_test) > 0.089491).astype(int)

pkl = 0.1 * price_vector.sum()
bau = Y_bau_pred.dot(price_vector)
stress = Y_stress_pred.dot(price_vector) ## эти переменные нужно записывать в таблицу с результатами

