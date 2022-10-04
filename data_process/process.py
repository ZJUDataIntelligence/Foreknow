import pandas as pd
import numpy as np
#data
a_data=pd.read_csv("application_train.csv")
b_data=pd.read_csv("bureau.csv")
c_data=pd.read_csv("credit_card_balance.csv")

#select
ad=a_data[['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',
'CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
'NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
'NAME_HOUSING_TYPE','CNT_FAM_MEMBERS']]

bd=b_data[['SK_ID_CURR','CREDIT_ACTIVE','CREDIT_CURRENCY','DAYS_CREDIT','AMT_CREDIT_MAX_OVERDUE',
'AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT',
'AMT_CREDIT_SUM_OVERDUE','CREDIT_TYPE','AMT_ANNUITY']]

cd=c_data[['SK_ID_CURR','MONTHS_BALANCE','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_ATM_CURRENT',
'AMT_DRAWINGS_CURRENT','AMT_DRAWINGS_OTHER_CURRENT','AMT_DRAWINGS_POS_CURRENT',
'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT',
'AMT_RECEIVABLE_PRINCIPAL','AMT_RECIVABLE','AMT_TOTAL_RECEIVABLE',
'CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_CURRENT','CNT_DRAWINGS_OTHER_CURRENT',
'CNT_DRAWINGS_POS_CURRENT','CNT_INSTALMENT_MATURE_CUM',
'NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']]

#delete
a=ad.dropna(axis=0, how='any')
b=bd.dropna(axis=0, how='any')
c=cd.dropna(axis=0, how='any')

#merge
df=pd.merge(a, b, on='SK_ID_CURR',how='inner')
df=pd.merge(df, c, on='SK_ID_CURR',how='inner')

#some classification attributes
df.loc[df["FLAG_OWN_CAR"] == "Y","FLAG_OWN_CAR"] = 1
df.loc[df["FLAG_OWN_CAR"] == "N","FLAG_OWN_CAR"] = 0
df.loc[df["FLAG_OWN_REALTY"] == "Y","FLAG_OWN_REALTY"] = 1
df.loc[df["FLAG_OWN_REALTY"] == "N","FLAG_OWN_REALTY"] = 0
df.loc[df["CREDIT_CURRENCY"] == "currency 1","CREDIT_CURRENCY"] = 1
df.loc[df["CREDIT_CURRENCY"] == "currency 2","CREDIT_CURRENCY"] = 2
df.loc[df["CREDIT_CURRENCY"] == "currency 3","CREDIT_CURRENCY"] = 3
df.loc[df["CREDIT_CURRENCY"] == "currency 4","CREDIT_CURRENCY"] = 4
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].astype(int)
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].astype(int)
df["CREDIT_CURRENCY"] = df["CREDIT_CURRENCY"].astype(int)

data_dummies = pd.get_dummies(df)
features = data_dummies.loc[:, 'SK_ID_CURR':'NAME_CONTRACT_STATUS_Signed']

#y
df['y_or_n']=df['SK_DPD'].map(lambda a : 1 if a>0 else 0)

#feature
df=df.drop('SK_DPD',axis=1)
feature=df.drop('SK_DPD_DEF',axis=1)

#three parts
ruan=feature[['SK_ID_CURR','NAME_CONTRACT_TYPE',
         'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY_x','AMT_GOODS_PRICE','CNT_FAM_MEMBERS', 'CODE_GENDER',
       'NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE']]
ruany=feature[['SK_ID_CURR','y_or_n']]#y
dai=feature[['SK_ID_CURR','AMT_BALANCE',
       'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
       'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
       'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
       'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
       'CNT_INSTALMENT_MATURE_CUM', 'NAME_CONTRACT_STATUS']]
li=feature[['SK_ID_CURR','CREDIT_CURRENCY', 'DAYS_CREDIT', 'AMT_CREDIT_MAX_OVERDUE',
       'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
       'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY_y','CREDIT_ACTIVE',
       'CREDIT_TYPE']]

ruan=ruan.drop_duplicates()
ruany=ruany.drop_duplicates()
dai=dai.drop_duplicates()
li=li.drop_duplicates()

ry=ruany.groupby("SK_ID_CURR").aggregate({'y_or_n':np.max}) #y:max

#d
d=dai.groupby("SK_ID_CURR").aggregate({'AMT_BALANCE' :np.mean, 'AMT_CREDIT_LIMIT_ACTUAL' :np.mean,
'AMT_DRAWINGS_ATM_CURRENT' :np.mean, 'AMT_DRAWINGS_CURRENT' :np.mean,
'AMT_DRAWINGS_OTHER_CURRENT' :np.mean, 'AMT_DRAWINGS_POS_CURRENT' :np.mean,
'AMT_INST_MIN_REGULARITY' :np.mean, 'AMT_PAYMENT_CURRENT' :np.mean,
'AMT_PAYMENT_TOTAL_CURRENT' :np.mean, 'AMT_RECEIVABLE_PRINCIPAL' :np.mean,
'AMT_RECIVABLE' :np.mean, 'AMT_TOTAL_RECEIVABLE' :np.mean, 'CNT_DRAWINGS_ATM_CURRENT' :np.mean,
'CNT_DRAWINGS_CURRENT' :np.mean, 'CNT_DRAWINGS_OTHER_CURRENT' :np.mean,
'CNT_DRAWINGS_POS_CURRENT' :np.mean, 'CNT_INSTALMENT_MATURE_CUM' :np.mean
})
linshi=feature[['SK_ID_CURR','NAME_CONTRACT_STATUS']]
linshi2=ruan['SK_ID_CURR']
linshi[linshi['NAME_CONTRACT_STATUS']=='Signed']='Active'
sd=linshi['NAME_CONTRACT_STATUS'].groupby(linshi['SK_ID_CURR']).value_counts().reset_index(name='count')
linshi3=sd[sd['NAME_CONTRACT_STATUS']=='Active'][['SK_ID_CURR','count']]
linshi3.columns = ['SK_ID_CURR', 'Active_count']
linshi3=linshi3.iloc[:9834,:]
linshi4=sd[sd['NAME_CONTRACT_STATUS']=='Completed'][['SK_ID_CURR','count']]
linshi4.columns = ['SK_ID_CURR', 'Completed_count']
linshi2=pd.merge(linshi2, linshi3, on='SK_ID_CURR',how='outer')
linshi2=pd.merge(linshi2, linshi4, on='SK_ID_CURR',how='outer')
linshi2=linshi2.fillna(0)
d=pd.merge(d,linshi2, on='SK_ID_CURR',how='inner')

#l
l=li.groupby("SK_ID_CURR").aggregate({'CREDIT_CURRENCY' :np.mean, 'DAYS_CREDIT' :np.min,
'AMT_CREDIT_MAX_OVERDUE' :np.max, 'AMT_CREDIT_SUM' :np.mean, 'AMT_CREDIT_SUM_DEBT' :np.mean,
'AMT_CREDIT_SUM_LIMIT' :np.mean, 'AMT_CREDIT_SUM_OVERDUE' :np.mean, 'AMT_ANNUITY_y' :np.mean
})
linshi=feature[['SK_ID_CURR','CREDIT_ACTIVE']]
linshi2=ruan['SK_ID_CURR']
sd=linshi['CREDIT_ACTIVE'].groupby(linshi['SK_ID_CURR']).value_counts().reset_index(name='count')
linshi3=sd[sd['CREDIT_ACTIVE']=='Active'][['SK_ID_CURR','count']]
linshi3.columns = ['SK_ID_CURR', 'Active_count']
linshi4=sd[sd['CREDIT_ACTIVE']=='Closed'][['SK_ID_CURR','count']]
linshi4.columns = ['SK_ID_CURR', 'Closed_count']
linshi2=pd.merge(linshi2, linshi3, on='SK_ID_CURR',how='outer')
linshi2=pd.merge(linshi2, linshi4, on='SK_ID_CURR',how='outer')
linshi2=linshi2.fillna(0)
l=pd.merge(l,linshi2, on='SK_ID_CURR',how='inner')
df=feature[['SK_ID_CURR','CREDIT_TYPE']]
sd=df['CREDIT_TYPE'].groupby(df['SK_ID_CURR']).value_counts().reset_index(name='count')
fi=sd.sort_values(['SK_ID_CURR', 'CREDIT_TYPE'],ascending=True).groupby('SK_ID_CURR').head(1)
fi.columns = ['SK_ID_CURR', 'CREDIT_TYPE_MAX', 'CREDIT_TYPE_max_count']
fi=fi[['SK_ID_CURR', 'CREDIT_TYPE_MAX']]
fi=fi.replace(['Microloan','Mortgage','Another type of loan','Cash loan (non-earmarked)','Loan for business development','Unknown type of loan'],['other','other','other','other','other','other'])
l=pd.merge(l,fi, on='SK_ID_CURR',how='inner')

#merge
df=pd.merge(ruan, d, on='SK_ID_CURR',how='inner')
df=pd.merge(df, l, on='SK_ID_CURR',how='inner')
df=pd.merge(df, ry, on='SK_ID_CURR',how='inner')

df.to_csv("one.csv") #9841 rows ¡Á 47 columns

#onehot
data_dummies1 = pd.get_dummies(ruan)
data_dummies2 = pd.get_dummies(d)
data_dummies3 = pd.get_dummies(l)
features = data_dummies.loc[:, 'SK_ID_CURR':'CREDIT_TYPE_MAX_other']
features1 = data_dummies1.loc[:, 'SK_ID_CURR':'NAME_HOUSING_TYPE_With parents']
features2 = data_dummies2.loc[:, 'SK_ID_CURR':'Completed_count']
features3 = data_dummies3.loc[:, 'SK_ID_CURR':'CREDIT_TYPE_MAX_other']

features.to_csv("binary.csv")#9841 rows ¡Á 74
features1.to_csv("r1_onehot.csv")#9841 rows ¡Á 40 columns
features2.to_csv("d1_onehot.csv")#9841 rows ¡Á 20 columns
features3.to_csv("l1_onehot.csv")#9841 rows ¡Á 15 columns