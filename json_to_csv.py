import numpy as np
import json
import os
import pathlib
import pandas as pd
import re

# Read annotation field of json files and extract relevant information to add to csv

dataset_directory = '/data/COVID/Data/covid-19-chest-x-ray-dataset'
json_directory = os.path.join(dataset_directory, 'releases/covid-only/annotations')

# Desired labels
Filename = []
AccessionID = []
Examination_Title = []
Age = []
Gender = []
Died = []

# New additions
ICU_scan = []
ICU_admission = []

# N/A for this dataset
SymptomOnset_DTM = []
Death_DTM = []


def replace_empty_with_missing(some_list):
    if len(some_list) == 0:
        some_list.append('Missing')
    return some_list


def tag_finder(some_list, tag):
    return replace_empty_with_missing([i for i in some_list if tag in i['name']])


def sex_survival_ICU_exam_label_converter(some_og_label, tag_identifier):
    some_og_label = str(some_og_label)
    if tag_identifier == 'Sex':
        if 'M' in some_og_label:
            return 'Male'
        elif 'F' in some_og_label:
            return 'Female'
        else:
            return some_og_label
    elif tag_identifier == 'Survival':
        if 'Y' in some_og_label:
            return 0.0
        elif 'N' in some_og_label:
            return 1.0
        else:
            return some_og_label
    elif tag_identifier == 'ICU':
        if 'Y' in some_og_label:
            return 1.0
        elif 'N' in some_og_label:
            return 0.0
        else:
            return some_og_label
    elif tag_identifier == 'Exam':
        if 'ray' in some_og_label:
            return 'X-ray'
        elif 'CT' in some_og_label:
            return 'CT'
        else:
            return some_og_label


icu_counter, surv_counter = 0, 0
tot_surv_labels = []
tot_icu_labels = []
tot_examination_labels = []
regex = re.compile(r'\d+')
for json_file in pathlib.Path(json_directory).rglob('*.json'):
    with open(os.path.join(json_directory, json_file)) as jf:
        # Load data and identify annotations
        data = json.load(jf)
        annotations = data['annotations']

        # Isolate individual labels
        surv = tag_finder(annotations, 'Survival')
        icu_admission = tag_finder(annotations, 'ICU_admission')
        icu_scan = tag_finder(annotations, 'ICU_scan')
        xray = tag_finder(annotations, 'ray')
        ct = tag_finder(annotations, 'CT')
        exam = xray + ct
        exam.remove('Missing')  # Either X-ray or CT, so this will always be a len 1 list
        age = tag_finder(annotations, 'Age')
        age = [replace_empty_with_missing(regex.findall(str(age[0])))[0]]
        if 'Missing' not in age:
            age[0] = np.int(age[0])
        sex = tag_finder(annotations, 'Sex')

        # Extend list of tags
        tot_surv_labels.extend(surv)

        # Increment counters
        surv_counter += len(surv)

        # Add to new dataframe lists
        # Filename.append(os.path.splitext(data['image']['filename'])[0])
        filename = os.path.splitext(data['image']['filename'])[0]
        original_filename = data['image']['original_filename']
        # Final filename of images is actually a combination of previous two fields, via underscore
        final_filename = '_'.join([filename, original_filename])
        Filename.append(final_filename)
        AccessionID.append('N/A')
        Examination_Title.append(sex_survival_ICU_exam_label_converter(exam[0], tag_identifier='Exam'))
        Age.append(age[0])
        Gender.append(sex_survival_ICU_exam_label_converter(sex[0], tag_identifier='Sex'))
        SymptomOnset_DTM.append('N/A')
        Death_DTM.append('N/A')
        Died.append(sex_survival_ICU_exam_label_converter(surv[0], tag_identifier='Survival'))
        ICU_scan.append(sex_survival_ICU_exam_label_converter(icu_scan[0], tag_identifier='ICU'))
        ICU_admission.append(sex_survival_ICU_exam_label_converter(icu_admission[0], tag_identifier='ICU'))

new_labels = pd.DataFrame({'Filename': Filename,
                           'AccessionID': AccessionID,
                           'Examination_Title': Examination_Title,
                           'Age': Age,
                           'Gender': Gender,
                           'SymptomOnset_DTM': SymptomOnset_DTM,
                           'Death_DTM': Death_DTM,
                           'Died': Died,
                           'ICU_scan': ICU_scan,
                           'ICU_admission': ICU_admission
                           })
# Ensure that filenames that are exclusively just numbers are kept intact
# Tells excel to treat the entry as a formula that returns the text within quotes
# new_labels.Filename = new_labels.Filename.apply('="{}"'.format)

new_labels.to_csv('covid-19-chest-x-ray-dataset_labels.csv')
