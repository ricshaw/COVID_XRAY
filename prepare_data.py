import pandas as pd
import numpy as np
import os
from datetime import date
pd.set_option('display.max_rows', 500)
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

## Load labs
#df_labs = pd.read_csv('/data/COVID/GSTT/labs_edit.csv')
df_labs = pd.read_csv('labs_edit.csv')
df_labs['CreatedWhen'] = pd.to_datetime(df_labs.CreatedWhen).dt.floor('1D')
#print(df_labs.head(200))

## Load data
#df = pd.read_csv('/data/COVID/GSTT/data.csv')
df = pd.read_csv('data.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.rename(columns={'PatientShortId': 'patient_pseudo_id'})

## Convert to spcecimen dates to datetime (Days)
df.SpecimenDate_1 = pd.to_datetime(df.SpecimenDate_1).dt.floor('1D')
df.SpecimenDate_2 = pd.to_datetime(df.SpecimenDate_2).dt.floor('1D')
df.SpecimenDate_3 = pd.to_datetime(df.SpecimenDate_3).dt.floor('1D')
df.SpecimenDate_4 = pd.to_datetime(df.SpecimenDate_4).dt.floor('1D')
df.SpecimenDate_5 = pd.to_datetime(df.SpecimenDate_5).dt.floor('1D')

## Repeat rows 5 times
df = df.loc[df.index.repeat(5)]

## Populate new column with all specimen dates
df.insert(1, 'CreatedWhen', '')
df.CreatedWhen = pd.to_datetime(df.CreatedWhen)
for i in range(5):
    df.loc[i::5,'CreatedWhen'] = df.loc[i::5,'SpecimenDate_%d' % (i+1)]
    df = df.reset_index(drop=True)

## Get columns for each dataset
eNot_cols = [c for c in df.columns if 'eNot' in c]
Sym_cols = [c for c in df.columns if 'Sym' in c]
EPR_cols = [c for c in df.columns if 'EPR' in c]
ICIP_cols = [c for c in df.columns if 'ICIP' in c]
eNot_cols.remove('eNot_SpecimenDate_Id')
eNot_cols.remove('eNot_ObsFound')
eNot_cols.remove('eNot_Height')
Sym_cols.remove('Sym_SpecimenDate_Id')
Sym_cols.remove('Sym_ObsFound')
EPR_cols.remove('EPR_SpecimenDate_Id')
EPR_cols.remove('EPR_ObsFound')
EPR_cols.remove('EPR_Age')
EPR_cols.remove('EPR_Gender')
EPR_cols.remove('EPR_Ethnicity')
EPR_cols.remove('EPR_Height')
ICIP_cols.remove('ICIP_SpecimenDate_Id')
ICIP_cols.remove('ICIP_Height')
all_cols = eNot_cols + Sym_cols + EPR_cols + ICIP_cols
print(Sym_cols)
print(EPR_cols)
print(ICIP_cols)
print(all_cols)

## Set rows != specimen_id to nan
for i in range(5):
    inds = df.loc[i::5, 'eNot_SpecimenDate_Id'] != (i+1)
    df.loc[inds[inds].index, eNot_cols] = np.nan
    inds = df.loc[i::5, 'Sym_SpecimenDate_Id'] != (i+1)
    df.loc[inds[inds].index, Sym_cols] = np.nan
    inds = df.loc[i::5, 'EPR_SpecimenDate_Id'] != (i+1)
    df.loc[inds[inds].index, EPR_cols] = np.nan
    inds = df.loc[i::5, 'ICIP_SpecimenDate_Id'] != (i+1)
    df.loc[inds[inds].index, ICIP_cols] = np.nan

## Drop null rows
df = df.drop(columns=['SpecimenDate_1', 'SpecimenDate_2', 'SpecimenDate_3', 'SpecimenDate_4', 'SpecimenDate_5'])
df = df[df['CreatedWhen'].notna()]
df = df.dropna(subset=all_cols, how='all')
df.to_csv('data_tmp.csv', index=False)

## Rename columns
df = df.rename(columns={'EPR_Age': 'Age',
                        'EPR_Gender': 'Gender',
                        'EPR_Ethnicity': 'Ethnicity',
                        'SpecimenDate_1': 'CreatedWhen'})

## Fix units
df['eNot_Height'] = df['eNot_Height'] * 100.0

## Combine columns with mean values
df['Height'] = df.loc[:, ['eNot_Height', 'MedChart_Height', 'EPR_Height', 'ICIP_Height']].mean(axis=1)
df['Weight'] = df.loc[:, ['eNot_Weight', 'MedChart_Weight', 'EPR_Weight', 'ICIP_Weight']].mean(axis=1)
df['BMI'] = df.loc[:, ['eNot_BMI','EPR_BMI', 'ICIP_BMI']].mean(axis=1)
df['NEWS2'] = df.loc[:, ['eNot_NEWS2', 'Sym_NEWS2']].mean(axis=1)
df['Temperature'] = df.loc[:, ['eNot_Temp', 'Sym_Temp']].mean(axis=1)
df['Urea'] = df['EPR_UreaLevel']
df['Sodium'] = df['EPR_SodiumLevel']
#df['Calcium'] = df['EPR_CorrectedCalciumLevel']
#df['Potassium'] = df['EPR_PotassiumLevel']
df['Systolic BP'] = df.loc[:, ['eNot_BPSys', 'Sym_BPSys']].mean(axis=1)
df['Diastolic BP'] = df.loc[:, ['eNot_BPDiast', 'Sym_BPDiast']].mean(axis=1)
df['Respiration Rate'] = df.loc[:, ['eNot_RespRate', 'Sym_RespRate']].mean(axis=1)
df['Bilirubin'] = df['EPR_BilirubinLevel']
df['Albumin'] = df['EPR_AlbuminLevel']
df['Alkaline Phosphatase'] = df['EPR_AlkPhosLevel']
df['Lymphocytes'] = df['EPR_Lymphocytes']
df['Heart Rate'] = df.loc[:, ['eNot_Pulse', 'Sym_Pulse']].mean(axis=1)
# df['GCS Score'] = df.loc[:, ['Sym_GCS']].mean(axis=1)
df['CRP'] = df['EPR_CRP']
df['Troponin'] = df['EPR_Troponin']
# df['PLT'] = df['EPR_PLT']
df['Neutrophils'] = df['EPR_Neutrophils']
df['GFR'] = df['EPR_EstimatedGFR']
df['Creatinine'] = df['EPR_CreatinineLevel']
df['ALT'] = df['EPR_AlanineTransLevel']

df['Died'] = df['EPR_DateOfDeath']
df['Died'] = np.where(df['Died'].isnull(), 0, 1)

# New additions
df[['eNot_SPO2', 'Sym_SPO2']] = df[['eNot_SPO2', 'Sym_SPO2']].apply(pd.to_numeric, errors='coerce')
df['Oxygen Saturation'] = df.loc[:, ['eNot_SPO2', 'Sym_SPO2']].mean(axis=1)
# See: https://www.ausmed.co.uk/cpd/articles/oxygen-flow-rate-and-fio2 for conversion
# FiO2 = ((FR x 100) + ((30 - FR) * 21)) / 30
df['FiO2'] = ((df['Sym_O2Amount'] * 100) + ((df['Respiration Rate'] - df['Sym_O2Amount']) * 21)) / df['Respiration Rate']
# Tricky to get a GCS score from this, see: https://learn.canvas.net/courses/2171/pages/assessment-of-neurological-functioning-acvpu
# Settle for finding average of each ACVPU condition on sym, and using this to populate eNot equivalent
temp = df.groupby('Sym_ACVPU', as_index=False)['Sym_GCS'].mean()
df.loc[df['eNot_ACVPU'] == 'Alert', 'eNot_ACVPU'] = float(temp[temp['Sym_ACVPU'] == 'Alert']['Sym_GCS'])
df.loc[df['eNot_ACVPU'] == 'New confusion', 'eNot_ACVPU'] = float(temp[temp['Sym_ACVPU'] == 'Confusion']['Sym_GCS'])
df.loc[df['eNot_ACVPU'] == 'Voice', 'eNot_ACVPU'] = float(temp[temp['Sym_ACVPU'] == 'Voice']['Sym_GCS'])
df.loc[df['eNot_ACVPU'] == 'Pain', 'eNot_ACVPU'] = float(temp[temp['Sym_ACVPU'] == 'Pain']['Sym_GCS'])

df[['Sym_GCS', 'eNot_ACVPU']] = df[['Sym_GCS', 'eNot_ACVPU']].apply(pd.to_numeric, errors='coerce')
df['GCS Score'] = df.loc[:, ['Sym_GCS', 'eNot_ACVPU']].mean(axis=1)

df[['eNot_BloodSugar', 'Sym_BloodSugar']] = df[['eNot_BloodSugar', 'Sym_BloodSugar']].apply(pd.to_numeric, errors='coerce')
df['Glu1'] = df.loc[:, ['eNot_BloodSugar', 'Sym_BloodSugar']].mean(axis=1)

df['clientvisit_admitdtm'] = df['ICIP_InTime']
df['clientvisit_dischargedtm'] = df['ICIP_OutTime']

print(df.Gender.unique())
print(df.Ethnicity.unique())
df['Gender'] = df['Gender'].replace('Male', 1)
df['Gender'] = df['Gender'].replace('Female', 0)

df.CreatedWhen = pd.to_datetime(df.CreatedWhen).dt.floor('1D')

df1 = df[['patient_pseudo_id', 
          'CreatedWhen', 
          'Gender', 
          'Age', 
          'Ethnicity', 
          'Height', 
          'Weight', 
          'BMI',
          'Albumin',
          'Alkaline Phosphatase',
          'ALT',
          'Bilirubin',
          'Creatinine',
          'CRP', 
          'Diastolic BP',
          'GCS Score',
          'GFR',
          'Heart Rate',
          'Lymphocytes',
          'Neutrophils',
          'NEWS2',
          #'PLT',
          'Respiration Rate',
          'Sodium',
          'Systolic BP',
          'Temperature',
          'Troponin',
          'Urea',
          'Oxygen Saturation',
          'FiO2',
          'Glu1',
          'Died',
          # 'clientvisit_admitdtm',
          # 'clientvisit_dischargedtm'
]]
print(df1.head(50))

## Merge with labs data
df1 = df1.reset_index(drop=True)
df_labs = df_labs.reset_index(drop=True)
df2 = pd.merge_asof(df1.sort_values('CreatedWhen'), df_labs.sort_values('CreatedWhen'), on='CreatedWhen', by='patient_pseudo_id')
df2 = df2.drop(columns='ESR')
df2.reset_index(drop=True)
df2 = df2.sort_values('patient_pseudo_id').reset_index(drop=True)
df2 = df2.rename(columns={'CreatedWhen': 'CXR_datetime'})
df2['Oxygen Saturation'] = df2.loc[:, ['Oxygen Saturation_x', 'Oxygen Saturation_y']].mean(axis=1)
df2 = df2.drop(columns=['Oxygen Saturation_x', 'Oxygen Saturation_y'])

## Fix ethnicity values
df2.Ethnicity = df2.Ethnicity.astype(str).apply(lambda x: 'White' if 'White' in x else x)
df2.Ethnicity = df2.Ethnicity.astype(str).apply(lambda x: 'Black' if 'Black' in x else x)
df2.Ethnicity = df2.Ethnicity.astype(str).apply(lambda x: 'Asian' if 'Asian' in x else x)
df2.Ethnicity = df2.Ethnicity.astype(str).apply(lambda x: 'Mixed' if 'Mixed' in x else x)
df2.Ethnicity = df2.Ethnicity.astype(str).apply(lambda x: 'Other' if 'Other' in x else x)
df2.Ethnicity = df2.Ethnicity.astype(str).apply(lambda x: 'Unknown' if 'Not Stated' in x else x)
print(df2.Ethnicity.unique())

## Remove any bad strings
for c in df2.columns:
    if not is_datetime(df2[c]) and c != 'Ethnicity':
        df2[c] = df2[c].astype(str).str.extract('(\d+)', expand=False).astype(np.float32)
        print(c, is_numeric_dtype(df2[c]))

## Sort by time
df2 = df2.groupby('patient_pseudo_id').apply(pd.DataFrame.sort_values, 'CXR_datetime')
df2 = df2.reset_index(drop=True)

## Fill missing
if True:
    df2 = df2.set_index('patient_pseudo_id').groupby(level='patient_pseudo_id').ffill().reset_index()
    df2 = df2.set_index('patient_pseudo_id').groupby(level='patient_pseudo_id').bfill().reset_index()
print(df2.head(20))

## Save
df2.to_csv('data_edit_new_extra.csv', index=False)
