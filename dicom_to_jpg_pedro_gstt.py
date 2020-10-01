import pandas as pd
import numpy as np
import pydicom
import os
import pathlib
import datetime
import glob
import cv2
import matplotlib.pytplot as plt

# images_directory = '/data/COVID/Data/KCH_news'  # '/data/COVID/Data/KCH_CXR'
images_directory = '/data/COVID/gstt/gstt_anon'
# original_labels = pd.read_csv('/data/COVID/Labels/KCH_CXR_Originals.csv')
original_labels = pd.read_csv('/data/COVID/GSTT/gstt.csv')
original_labels = original_labels.sort_values(by='patient_pseudo_id')
# Remove duplicates: Will be parsing through directories according to subID: Contains all scans to no need to do this >1
original_labels = original_labels.drop_duplicates(subset=['patient_pseudo_id'], ignore_index=True)
latest_flag = False

if latest_flag:
    new_images_directory = '/data/COVID/GSTT_JPGs_latest'
else:
    new_images_directory = '/data/COVID/GSTT_JPGs_All'

if not os.path.exists(new_images_directory):
    os.makedirs(new_images_directory)


# Correction functions
def correct_monochrome(image):
    return image.max() - image


def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """

    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1] - 1)

    data_new[data <= (wl - ww / 2.0)] = out_range[0]
    data_new[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] = \
        ((data[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] - (wl - 0.5)) / (ww - 1.0) + 0.5) * (
                    out_range[1] - out_range[0]) + out_range[0]
    data_new[data > (wl + ww / 2.0)] = out_range[1] - 1

    return data_new.astype(dtype)


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
for iterator, subID in enumerate(original_labels['patient_pseudo_id']):
    print(subID)
    # Want to find corresponding DICOM file, load it, pre-process, save as JPG
    # Find dicoms contained therein
    dicoms = [f for f in pathlib.Path(images_directory).rglob(f'{subID}_*.dcm')]
    # print(dicoms)
    # Load in dicom
    if not dicoms:
        print(f'Missing the following subject in images directory: {subID}')
        missing_subs.append(subID)
        continue
    scan_times_list = []
    for num, dicom in enumerate(dicoms):
        # try:
        dicom_object = pydicom.dcmread(dicom, force=True)
        # try:
        # img = dicom_object.pixel_array.astype(np.float32)
        if "AcquisitionDate" in dicom_object:
            scan_time = dicom_object.AcquisitionDate
        elif "ContributionDate" in dicom_object:
            scan_time = dicom_object.ContributionDate
        scan_time = np.int(float(scan_time))
        # Append to scan times list to later determine "latest" scan and only save that one for this subject
        scan_times_list.append(scan_time)

    # Only if this condition is met did things pass
    if latest_flag:
        if min(scan_times_list) > 0:
            latest_scan_index = scan_times_list.index(max(scan_times_list))
            latest_dicom = dicoms[latest_scan_index]
            latest_dicom_object = pydicom.dcmread(latest_dicom, force=True)
            try:
                # img = latest_dicom_object.pixel_array.astype(np.float32)
                wl, ww = latest_dicom_object.WindowCenter, latest_dicom_object.WindowWidth
                if isinstance(wl, pydicom.multival.MultiValue):
                    arr = win_scale(data=latest_dicom_object.pixel_array, wl=int(wl[0]),
                                    ww=int(ww[0]), dtype=np.float32, out_range=[0, 16383])
                else:
                    arr = win_scale(data=latest_dicom_object.pixel_array, wl=int(wl),
                                    ww=int(ww), dtype=np.float32, out_range=[0, 16383])
                # Correct monochromes
                if 'PhotometricInterpretation' in latest_dicom_object:
                    if latest_dicom_object.PhotometricInterpretation == 'MONOCHROME1':
                        print(subID, 'Mono', latest_dicom)
                        arr = correct_monochrome(arr)

                # Normalising
                # arr -= np.min(arr)
                arr /= np.max(arr)
                arr = np.uint8(255.0 * arr)

                print(latest_dicom_object.AccessionNumber)

                if "AcquisitionDate" in latest_dicom_object:
                    scan_time = latest_dicom_object.AcquisitionDate
                elif "ContributionDate" in latest_dicom_object:
                    scan_time = latest_dicom_object.ContributionDate
                scan_time = np.int(float(scan_time))

                # Append to scan times list to later determine "latest" scan and only save that one for this subject
                scan_times_list.append(scan_time)
                filename = '_'.join([str(subID), str(scan_time) + '.jpg'])

                # Save image
                print('Saving image!')
                cv2.imwrite('/'.join([new_images_directory, f'{filename}']), arr)
            except:
                print('Oops! No image data')
    else:
        for num, dicom in enumerate(dicoms):
            # try:
            latest_dicom_object = pydicom.dcmread(dicom, force=True)
            try:
                # img = latest_dicom_object.pixel_array.astype(np.float32)
                wl, ww = latest_dicom_object.WindowCenter, latest_dicom_object.WindowWidth
                if isinstance(wl, pydicom.multival.MultiValue):
                    arr = win_scale(data=latest_dicom_object.pixel_array, wl=int(wl[0]),
                                    ww=int(ww[0]), dtype=np.float32, out_range=[0, 16383])
                else:
                    arr = win_scale(data=latest_dicom_object.pixel_array, wl=int(wl),
                                    ww=int(ww), dtype=np.float32, out_range=[0, 16383])
                # Correct monochromes
                if 'PhotometricInterpretation' in latest_dicom_object:
                    if latest_dicom_object.PhotometricInterpretation == 'MONOCHROME1':
                        print(subID, 'Mono', dicom)
                        arr = correct_monochrome(arr)

                # Normalising
                # arr -= np.min(arr)
                arr /= np.max(arr)
                arr = np.uint8(255.0 * arr)

                if "AcquisitionDate" in latest_dicom_object:
                    scan_time = latest_dicom_object.AcquisitionDate
                elif "ContributionDate" in latest_dicom_object:
                    scan_time = latest_dicom_object.ContributionDate
                scan_time = np.int(float(scan_time))

                # Append to scan times list to later determine "latest" scan and only save that one for this subject
                scan_times_list.append(scan_time)
                filename = '_'.join([str(subID), str(scan_time) + '.jpg'])

                # Save image
                print('Saving image!')
                cv2.imwrite('/'.join([new_images_directory, f'{filename}']), arr)
            except:
                print('Oops! No image data')

post_process = True
counts = 0
relevant_indices = []
if post_process:
    # Loop through images and create tag in csvs
    # For all entries with given patient_pseudo_id, collect CXR dates from csvs
    original_labels = pd.read_csv('/data/COVID/GSTT/gstt.csv')
    # Empty fill
    original_labels['Filename'] = ""
    original_labels['Time_Mismatch'] = 500
    unique_patient_ids = original_labels.patient_pseudo_id.unique()
    all_images_directory = '/data/COVID/GSTT_JPGs_All'
    for id in unique_patient_ids:
        patient_series = original_labels[original_labels.patient_pseudo_id == id]
        # print(patient_series.index)
        patient_times = patient_series.CXR_datetime
        patient_times = [x.replace("-", "") for x in patient_times]
        date_list_converted = [datetime.datetime.strptime(each_date, "%Y%m%d").date() for each_date in patient_times]
        # Find image matches to patient pseudo id
        for img_name in glob.glob(os.path.join(all_images_directory, f'{id}_*')):
            extracted_time_string = os.path.basename(img_name).split('_')[1].split('.')[0]
            extracted_time = datetime.datetime.strptime(extracted_time_string, "%Y%m%d").date()
            # Date proximity: https://stackoverflow.com/questions/54632081/comparing-dates-and-finding-the-closest-date-to-the-current-date
            differences = [abs(extracted_time - each_date) for each_date in date_list_converted]
            minimum = min(differences)
            closest_date_index = differences.index(minimum)
            # Find index of relevant entry
            relevant_index = patient_series.index[closest_date_index]
            relevant_indices.append(relevant_index)
            # Update dataframe: Only if mismatch is smaller! Otherwise might overwrite with worse image!
            if minimum.days < original_labels.at[relevant_index, 'Time_Mismatch']:
                original_labels.at[relevant_index, 'Filename'] = os.path.basename(img_name)
                original_labels.at[relevant_index, 'Time_Mismatch'] = minimum.days
            # else:
            #     original_labels.at[relevant_index, 'Filename'] = img_name
            #     original_labels.at[relevant_index, 'Time_Mismatch'] = minimum.days
            print(relevant_index)
            counts += 1

# Plot histogram of time mismatches
original_labels['Time_Mismatch'] = original_labels['Time_Mismatch'].astype(float)
original_labels['Time_Mismatch'].plot.hist(by='Time_Mismatch', bins=40)
plt.show()

# Value counts
viable_entries = original_labels.Time_Mismatch.value_counts()[0] + original_labels.Time_Mismatch.value_counts()[1]
print(f'Number of viable entries is {viable_entries}')

# Make invalid mismatches nulls
original_labels.loc[original_labels.Time_Mismatch == 500, 'Time_Mismatch'] = np.nan
original_labels.loc[original_labels.Filename == "", 'Filename'] = np.nan
original_labels.to_csv(os.path.join('/data/COVID/Labels', 'new_gstt.csv'), index=False)
# Better approach would find combination of dates that minimises overall difference
