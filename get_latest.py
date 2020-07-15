import numpy as np
import pandas as pd
from datetime import datetime as dt

# Load full list
df = pd.read_csv('KCH_CXR_JPG2.csv')
print(df.shape)

# Get unique patients
labels = df.drop_duplicates(subset=['AccessionID'], ignore_index=True)
print('Unique patients', labels.shape)

df_out = pd.DataFrame()
# Loop through unique patients
for l in labels.AccessionID:
    fs = []
    times = []
    rows = []
    # Get all matches of that patient
    for i, f in enumerate(df.Filename):
        if l in f:
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
    print(fs)
    print(times)
    # Pick latest
    id = np.argmax(times)
    print(id)
    df_out = df_out.append(rows[id])
    #else:
    #    df_out = df_out.append(row)

print('Out', df_out.shape)
df_out = df.drop_duplicates(subset=['AccessionID'], ignore_index=True)
print('Unique out', df_out.shape)
df_out.to_csv('KCH_CXR_JPG2_latest.csv', index=False)
