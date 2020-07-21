import numpy as np
import pandas as pd
from datetime import datetime as dt
import os

# Load full list
#df = pd.read_csv('KCH_CXR_JPG.csv')
df = pd.read_csv('all_jpgs_unique.csv')
print(df.shape)

# Get labels
labels = pd.read_csv('cxr_news2_pseudonymised.csv')
labels["Filename"] = ""

count = 0
#df_out = pd.DataFrame()
# Loop through accessions
for index, l in enumerate(labels.Accession):
    print('\nAccession:', l)
    fs = []
    times = []
    rows = []
    # Get all matches of that accessionID
    for i, f in enumerate(df.Filename):
        if l in f:
            print(f)
            count += 1
            fs.append(f)
            name, datetime = f.split('_')
            year = datetime[0:4]
            month = datetime[4:6]
            day = datetime[6:8]
            time = datetime[8::]
            hour = time[:2]
            min = time[2:4]
            sec = time[4:6]
            tt = dt.strptime(month+'/'+day+'/'+year+'/'+hour+'/'+min+'/'+sec, "%m/%d/%Y/%H/%M/%S")
            times.append(tt)
            row = df.iloc[i]
            rows.append(row)
    #print(fs)
    print(times)
    # Pick latest
    if len(times)>0:
        id = np.argmax(times)
        print(id)
        #df_out = df_out.append(rows[id])
        labels["Filename"].iloc[index] = os.path.join('/nfs/project/covid/CXR/KCH_CXR_JPG', fs[id])
    #else:
    #    df_out = df_out.append(row)
print(count)
#print('Out', df_out.shape)
#print(df_out.head())

#df_out = df.drop_duplicates(subset=['AccessionID'], ignore_index=True)
#print('Unique out', df_out.shape)
#df_out.to_csv('KCH_CXR_JPG2_latest.csv', index=False)
print(labels.shape)
print(labels.head())
labels.to_csv('cxr_news2_pseudonymised_filenames.csv', index=False)
