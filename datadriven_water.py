# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', sep = ',')
target = pd.read_csv('target.csv', sep = ',')
test = pd.read_csv('test.csv', sep = ',')

train.head(10)
colnames = train.columns
'''
Index(['id', 'amount_tsh', 'date_recorded', 'funder', 'gps_height',
       'installer', 'longitude', 'latitude', 'wpt_name', 'num_private',
       'basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga',
       'ward', 'population', 'public_meeting', 'recorded_by',
       'scheme_management', 'scheme_name', 'permit', 'construction_year',
       'extraction_type', 'extraction_type_group', 'extraction_type_class',
       'management', 'management_group', 'payment', 'payment_type',
       'water_quality', 'quality_group', 'quantity', 'quantity_group',
       'source', 'source_type', 'source_class', 'waterpoint_type',
       'waterpoint_type_group'],
      dtype='object')
'''
train.shape
#(59400,40)
test.shape
#(14850,40)
target.shape
#[Id, status_group], both objects

target['status_group'].value_counts()
'''
functional                 32259
non functional             22824
functional needs repair     4317
Name: status_group, dtype: int64
'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target['status_group'] = le.fit_transform(target['status_group'])
'''
0 = functional
1 = non-functional
2 = functional needs repair
'''

#check for correlation between target and predictors
target_corr = list()

for c, v in enumerate(train, start = 1):
    target_corr.append(target.corrwith(train[v], method = 'spearman'))
    
target_corr = pd.DataFrame(data = target_corr, index = train.columns, columns = ['correlation'])
target_corr = target_corr[target_corr['correlation']!=0]
target_corr = target.corr.sort_values(ascending = False)


#combine data
combine = pd.concat([train, test]), axis =0)

#look for missing values
miss_vals = pd.Series(combine.isnull().sum(), name = 'PctMissing')
miss_vals = miss_vals[miss_vals!=0]
miss_vals = miss_vals.sort_values(ascending = False)
print(miss_vals)
'''
scheme_name          35258
scheme_management     4846
installer             4532
funder                4504
public_meeting        4155
permit                3793
subvillage             470
Name: PctMissing, dtype: int64
'''
#pct missing
miss_vals / len (combine)
'''
scheme_name          0.474855
scheme_management    0.065266
installer            0.061037
funder               0.060660
public_meeting       0.055960
permit               0.051084
subvillage           0.006330
Name: PctMissing, dtype: float64
'''

#subvillage could just use mode of region
combine['subvillage'] = combine.groupby('region')['subvillage'].transform(lambda x:x.fillna(x.mode()[0]))

#check the value counts of 'permit'
'''
combine['permit'].value_counts()
True     48606
False    21851
Name: permit, dtype: int64
combine['permit'].isnull().sum()
Out[23]: 3793
'''
