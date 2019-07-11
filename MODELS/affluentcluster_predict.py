
################################    affluentcluster_predict.py   ###############################

import argparse
from time import gmtime, strftime
parser = argparse.ArgumentParser(description='MLM Environment Variables')
parser.add_argument('--login', default='U_001A5')
parser.add_argument('--password', default='')
parser.add_argument('--tns', default="""(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=dsacrmap)(PORT=1521)))(CONNECT_DATA=(SID=DMMLMPLT)(SERVER=DEDICATED)))""")
parser.add_argument('--job', default=strftime("%y%m%d%H%M%S9900002", gmtime()))
args = vars(parser.parse_known_args()[0])
#print("JOB:"+args["job"]+" Login:"+args["login"]+" Pwd:"+args["password"]+" TNS:"+args["tns"])

import numpy as np
import pandas as pd
import cx_Oracle
import datetime
#import pickle
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier

###### Get connection
conn_bi = cx_Oracle.connect(args["login"],args["password"], args["tns"])
cursor_bi = conn_bi.cursor()

###### Get Parameters (Report Month)
sqlRepMonth = """select to_char(to_date(param_value,'yymm'),'yyyy-mm-dd' ) PARAM_VALUE from param where param_name='date' and model_name='AffluentClusterModel'
            """
dReportDateYYYYMMDD = pd.read_sql(sqlRepMonth, conn_bi)["PARAM_VALUE"][0]
dReportDateYYMM = datetime.datetime.strptime(dReportDateYYYYMMDD, '%Y-%m-%d').strftime('%y%m')


###### Prediction

cluster_centers = [[  1.00000000e+00,   8.93401950e-04,   4.45549684e-05,
          2.09623554e-03,   3.75526440e-03,   3.58592594e-01,
          1.27977178e-03,   1.62085350e-01,   5.34329776e-04],
       [  9.27643785e-05,   2.33614203e-01,   1.43654917e-01,
          3.20839874e-01,   3.80185001e-02,   5.76457275e-02,
          2.11341350e-01,   4.13278558e-01,   9.61158825e-01],
       [  5.35303250e-05,   1.71806173e-01,   2.47684813e-02,
          2.44526524e-01,   1.83565566e-02,   1.41483715e-01,
          6.17259711e-01,   3.75867000e-01,   8.56868654e-03],
       [  1.66255898e-14,   7.65352599e-01,   3.85679883e-02,
          5.73947499e-01,   9.47164014e-02,   2.14533849e-01,
          1.40187273e-01,   3.05339975e-01,   5.12288716e-03],
       [  5.81564408e-05,   2.68985170e-01,   3.90276243e-02,
          3.09114904e-01,   5.27436088e-01,   8.85951669e-01,
          8.11513596e-02,   3.89972168e-01,   9.20009305e-03],
       [ -1.22402088e-14,   1.20400353e-01,   7.34823864e-01,
          5.58088585e-01,   5.66573748e-02,   1.18229690e-01,
          1.00070065e-01,   3.31216620e-01,   1.07252968e-02],
       [ -4.58522109e-14,   6.80229925e-02,   2.35051626e-02,
          1.52084465e-01,   2.93427208e-02,   1.16196756e-01,
          1.09634524e-13,   2.24275511e-01,   1.11640763e-02]]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(cluster_centers, range(0,7)) 

sqlStr_bi = """SELECT ONLY_DEP,CN_DC_3,CN_CREDIT_3,POP_3,REST_CV_SAVE_AVG, REST_ALL_AVG,ZP_BI,
                                  CNT_PRODUCT,OPLATA_CRED_SUM,PIN_EQ FROM output_""" + dReportDateYYMM 
#sqlStr_bi += "  WHERE REPORT_DATE = to_date('"+dReportDateDDMMYYYY+"','dd.mm.yyyy')"
            
df_bi_nov = pd.read_sql(sqlStr_bi, conn_bi)

df_bi_nov = df_bi_nov.fillna(0)

df_bi_nov_scaled = df_bi_nov[["ONLY_DEP","CN_DC_3","CN_CREDIT_3","POP_3","REST_CV_SAVE_AVG", "REST_ALL_AVG","ZP_BI",
                                  "CNT_PRODUCT","OPLATA_CRED_SUM","PIN_EQ"]].copy()
min_max_scaler = preprocessing.MinMaxScaler()
for i in range(1,9):
    df_bi_nov_scaled[df_bi_nov_scaled.columns[i]]=min_max_scaler.fit_transform(df_bi_nov_scaled[df_bi_nov_scaled.columns[i]].astype(float).reshape((-1,1)))
    
bi_nov_predicted = neigh.predict(df_bi_nov_scaled[["ONLY_DEP","CN_DC_3","CN_CREDIT_3","POP_3","REST_CV_SAVE_AVG", "REST_ALL_AVG","ZP_BI","CNT_PRODUCT","OPLATA_CRED_SUM"]])

###### Write Results to OUTPUT table

out_bi = df_bi_nov[["PIN_EQ", "ONLY_DEP","CN_DC_3","CN_CREDIT_3","POP_3","REST_CV_SAVE_AVG", "REST_ALL_AVG","ZP_BI","CNT_PRODUCT","OPLATA_CRED_SUM"]].copy()
out_bi["CLUSTER_BI"] = bi_nov_predicted

cursor_bi.execute("update AFFLCLUSTER_OUT set ACTIVE = 0 where ACTIVE = 1 and REPORT_DATE = date'"""+dReportDateYYYYMMDD+"'")
cursor_bi.prepare("""insert into AFFLCLUSTER_OUT (PIN_EQ, ONLY_DEP, CN_DC_3,CN_CREDIT_3,POP_3,REST_CV_SAVE_AVG, REST_ALL_AVG,ZP_BI,CNT_PRODUCT,OPLATA_CRED_SUM,
                                                  CLUSTER_BI, ACTIVE, REPORT_DATE, JOB) 
                  values (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, 1, date'"""+dReportDateYYYYMMDD+"', "+args["job"]+")")
cursor_bi.executemany(None, out_bi.values.tolist())
conn_bi.commit()

conn_bi.close()