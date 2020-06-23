import pandas as pd
import pydicom
import numpy as np
import os
import cv2
from pathlib import Path


#csv = [f for f in Path(r'./').rglob('*.png')]
#csv = [f for f in Path(r'./').rglob('*.jpg')]
#csv = [f for f in Path(r'./').rglob('*.dcm')]
#files = []
#for f in csv:
#    files.append(os.fspath(f.absolute()))
#df = pd.DataFrame({'filepath':files})
#df.to_csv('out.csv')

filepaths = 'KCH_CXR.csv'
#filepaths = 'PUBLIC.csv'
df = pd.read_csv(filepaths)
print(df.shape)
print(df.head())

save_path = '/nfs/project/richard/COVID/KCH_CXR_JPG'
filelist = []
for f in df.filepath:
    #print(f)
    head, tail = os.path.split(f)
    tail = os.path.splitext(tail)[0]
    #print(tail)
    ds = pydicom.dcmread(f)
    try:
        img = ds.pixel_array.astype(np.float32)
        img -= img.min()
        img /= img.max()
        img = np.uint8(255.0*img)
        save_name = os.path.join(save_path, tail+'.jpg')
        cv2.imwrite(save_name, img)
        filelist.append(save_name)
    except:
        print('Cannot load image')

df = pd.DataFrame({'filepath':filelist})
df.to_csv('KCH_CXR_JPG.csv')
