# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
#import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
#import data
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

#%%
#look at train / test shapes
train.shape
train_shape = train.shape[0]
#(59400,40)
test.shape
#(14850,40)
target.shape
#[Id, status_group], both objects

#%%
#look at the value_counts of the target variable
target['status_group'].value_counts()
'''
functional                 32259
non functional             22824
functional needs repair     4317
Name: status_group, dtype: int64
'''

#%% 
#LabelEncode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target['status_group'] = le.fit_transform(target['status_group'])
'''
0 = functional
1 = non-functional
2 = functional needs repair
'''

#%%
#look at correlations of variables with target
#check for correlation between target and predictors
target_corr = list()

for c, v in enumerate(train, start = 1):
    target_corr.append(target.corrwith(train[v], method = 'spearman'))
    
target_corr = pd.Series(data = target_corr, index = train.columns, name = 'correlation')
target_corr = abs(target_corr)

#%%
#combine data
combine = pd.concat([train, test],  axis = 0).reset_index(drop = True)

#%%    
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

#%%
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

#%%
#these variables all seem to be about area / region. Might be best to use the mode of the region they're in
combine['subvillage'] = combine.groupby('region')['subvillage'].transform(lambda x:x.fillna(x.mode()[0]))
combine['public_meeting'] = combine.groupby('region')['public_meeting'].transform(lambda x:x.fillna(x.mode()[0]))
combine['permit'] = combine.groupby('region')['permit'].transform(lambda x:x.fillna(x.mode()[0]))
combine['funder'] = combine.groupby('region')['funder'].transform(lambda x:x.fillna(x.mode()[0]))
combine['installer'] = combine.groupby('region')['funder'].transform(lambda x:x.fillna(x.mode()[0]))
combine['scheme_management'] = combine.groupby('region')['scheme_management'].transform(lambda x:x.fillna(x.mode()[0]))

#%%
#scheme_management and scheme_name seem to be redundant. Lots of missing values for scheme_name. Will delete.
combine = combine.drop(['scheme_name'], axis = 1)

#%%
#check missing values
miss_vals = pd.Series(combine.isnull().sum(), name = 'PctMissing')
miss_vals = miss_vals[miss_vals!=0]
miss_vals = miss_vals.sort_values(ascending = False)
print(miss_vals)

#values all filled in.

#%%
dtyp = pd.Series(combine.dtypes, name = 'dtype')

#%%
#some items as int are actually objects
combine['construction_year'] = combine['construction_year'].astype('object')

#%%
#check correlations between variables. See if there's some that can be deleted
f = plt.figure(figsize=(19, 15))
plt.matshow(combine.corr(), fignum=f.number)
plt.xticks(range(combine.select_dtypes(['number']).shape[1]), combine.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(combine.select_dtypes(['number']).shape[1]), combine.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

corr_df = combine.corr()
#correlations are all very low. 
corr_target = combine.corrwith(target)