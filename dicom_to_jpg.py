import pandas as pd
import pydicom
import numpy as np
import os
import cv2
from pathlib import Path


labels = pd.read_csv('KCH_CXR_labels.csv')
print(labels.shape)

PATH = '/nfs/project/covid/CXR/KCH_CXR'
save_path = '/nfs/project/richard/COVID/KCH_CXR_JPG'

Filename = []
AccessionID = []
Examination_Title = []
Age = []
Gender = []
SymptomOnset_DTM = []
Death_DTM = []
Died = []

for i, name in enumerate(labels.AccessionID):
    print(i, name)
    img_dir = os.path.join(PATH, name)
    files = [f for f in Path(img_dir).rglob('*.dcm')]
    #print(files)

    for f in files:
        print(os.fspath(f.absolute()))
        ds = pydicom.dcmread(f)
        try:
            img = ds.pixel_array.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img = np.uint8(255.0*img)
            print(img.shape)

            name = name + '_' + ds.AcquisitionDateTime.split('.')[0]
            save_name = os.path.join(save_path, (name + '.jpg'))
            print(save_name)
            cv2.imwrite(save_name, img)

            Filename.append(name)
            AccessionID.append(labels.AccessionID[i])
            Examination_Title.append(labels.Examination_Title[i])
            Age.append(labels.Age[i])
            Gender.append(labels.Gender[i])
            SymptomOnset_DTM.append(labels.SymptomOnset_DTM[i])
            Death_DTM.append(labels.Death_DTM[i])
            Died.append(labels.Died[i])

        except:
            print('Cannot load image')

df = pd.DataFrame({'Filename':Filename,
                   'AccessionID':AccessionID,
                   'Examination_Title':Examination_Title,
                   'Age':Age,
                   'Gender':Gender,
                   'SymptomOnset_DTM':SymptomOnset_DTM,
                   'Death_DTM':Death_DTM,
                   'Died':Died
                    })
df.to_csv('KCH_CXR_JPG.csv')
exit(0)







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
