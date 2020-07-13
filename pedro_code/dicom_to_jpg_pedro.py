import pandas as pd
import numpy as np
import pydicom
import os
import pathlib
from datetime import datetime
import cv2

images_directory = '/data/COVID/Data/KCH_CXR'
original_labels = pd.read_csv('/data/COVID/Labels/KCH_CXR_Originals.csv')
original_labels = original_labels.sort_values(by='AccessionID')
# Remove duplicates: Will be parsing through directories according to subID: Contains all scans to no need to do this >1
original_labels = original_labels.drop_duplicates(subset=['AccessionID'], ignore_index=True)

new_images_directory = '/data/COVID/pedro_images_latest_additions'
if not os.path.exists(new_images_directory):
    os.makedirs(new_images_directory)

Filename = []
AccessionID = []
Examination_Title = []
Age = []
Gender = []
SymptomOnset_DTM = []
Death_DTM = []
Died = []

missing_subs = []
bad_dicoms = []
counter = 0
# CSV file creation
for iterator, subID in enumerate(original_labels.AccessionID):
    print(iterator, subID)
    # Want to find corresponding DICOM file, load it, pre-process, save as JPG
    image_dir = os.path.join(images_directory, subID)
    # Find dicoms contained therein
    dicoms = [f for f in pathlib.Path(image_dir).rglob('*.dcm')]
    # Load in dicom
    if not dicoms:
        print(f'Missing the following subject in images directory: {subID}')
        missing_subs.append(subID)
        continue
    scan_times_list = []
    for num, dicom in enumerate(dicoms):
        dicom_object = pydicom.dcmread(dicom)
        try:
            img = dicom_object.pixel_array.astype(np.float32)
            if "AcquisitionDateTime" in dicom_object:
                scan_time = dicom_object.AcquisitionDateTime
            elif "ContributionDateTime" in dicom_object:
                scan_time = dicom_object.ContributionDateTime
            scan_time = np.int(float(scan_time))
            # Append to scan times list to later determine "latest" scan and only save that one for this subject
            scan_times_list.append(scan_time)
        except:
            print('Error loading file!')
            scan_times_list.append(-1e40)
            bad_dicoms.append('_'.join([subID, f'_{num}']))
    # Only if this condition is met did things pass
    if (min(scan_times_list) > 0) and not original_labels.iloc[iterator].isnull().any():
        latest_scan_index = scan_times_list.index(max(scan_times_list))
        latest_dicom = dicoms[latest_scan_index]
        latest_dicom_object = pydicom.dcmread(latest_dicom)
        img = latest_dicom_object.pixel_array.astype(np.float32)
        # elif len(scan_times_list) == 1:
        #     img = dicom_object.pixel_array.astype(np.float32)
        img -= np.min(img)
        img /= np.max(img)
        img = np.uint8(255.0 * img)

        # Isolate date and time of onset
        time_of_death = datetime.strptime(original_labels.iloc[iterator].Death_DTM, '%d/%m/%Y')
        if time_of_death.year < 2000:
            died = 0.0
        else:
            died = 1.0
        if "AcquisitionDateTime" in dicom_object:
            scan_time = dicom_object.AcquisitionDateTime
        elif "ContributionDateTime" in dicom_object:
            scan_time = dicom_object.ContributionDateTime
        scan_time = np.int(float(scan_time))
        # Append to scan times list to later determine "latest" scan and only save that one for this subject
        scan_times_list.append(scan_time)
        filename = '_'.join([subID, str(scan_time) + '.jpg'])
        # Add to new dataframe lists
        Filename.append(filename)
        AccessionID.append(subID)
        Examination_Title.append(original_labels.Examination_Title[iterator])
        Age.append(original_labels.Age[iterator])
        Gender.append(original_labels.Gender[iterator])
        SymptomOnset_DTM.append(original_labels.SymptomOnset_DTM[iterator])
        Death_DTM.append(original_labels.Death_DTM[iterator])
        Died.append(died)

        # Save image
        cv2.imwrite('/'.join([new_images_directory, f'{filename}']), img)

new_labels = pd.DataFrame({'Filename': Filename,
                           'AccessionID': AccessionID,
                           'Examination_Title': Examination_Title,
                           'Age': Age,
                           'Gender': Gender,
                           'SymptomOnset_DTM': SymptomOnset_DTM,
                           'Death_DTM': Death_DTM,
                           'Died': Died
                           })
print(missing_subs)
print(bad_dicoms)
new_labels.to_csv('/data/COVID/Labels/KCH_CXR_JPG_latest_latest_additions.csv')
