import pandas as pd
import numpy as np
import os
from datetime import date


## Load data
df = pd.read_csv('/data/COVID/GSTT/labs.csv')
df = df.replace('Ferritin COVID-19', 'Ferritin')
df = df.replace('D-Dimers', 'D-Dimer')
df.Value = pd.to_numeric(df.Value, errors='coerce')
df = df.drop(columns=['UnitOfMeasure'])
pids = df.PatientShortId.unique()
cols = df.ItemName.unique()
print(cols)

## Resample timestamps
df['CreatedWhen'] = pd.to_datetime(df.CreatedWhen)
df = df.groupby('PatientShortId').apply(pd.DataFrame.sort_values, 'CreatedWhen')
df['CreatedWhen'] = df['CreatedWhen'].dt.floor('1D')
#df['CreatedWhen'] = df['CreatedWhen'].dt.floor('1h')
#df = df.drop_duplicates(['PatientShortId', 'CreatedWhen', 'ItemName'])
df = df.reset_index(drop=True)
df = df.groupby(['PatientShortId', 'CreatedWhen', 'ItemName'], as_index=False)['Value'].mean()
df = df.reset_index(drop=True)
print(df.head(40))

## Reorder data
df_cols = []
for c in cols:
    df_c = df[df['ItemName'].str.match(c)]
    df_c = df_c.drop(columns=['ItemName'])
    df_c = df_c.rename(columns={'Value': c})
    df_c[c] = df_c[c].astype(np.float32)
    df_c = df_c.reset_index(drop=True)
    df_cols.append(df_c)
    print(df_c.head())

## Merge dataframes
df1 = df_cols[0]
for i in range(1,len(df_cols)):
    df1 = pd.merge_ordered(df1, df_cols[i], fill_method=None, left_by="PatientShortId")
    print(df1.head())

## Fix values
df1['D-Dimer'] = df1['D-Dimer'] * 1000.0
df1 = df1.rename(columns={'PatientShortId': 'patient_pseudo_id'})
print(df1.head(40))

## Save
df1.to_csv('/data/COVID/GSTT/labs_edit.csv', index=False)
