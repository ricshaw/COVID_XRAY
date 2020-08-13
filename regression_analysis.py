import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

reg_no_imaging = '/data/COVID/models/reg-adaptive-no-imaging'
reg_imaging_bloods = '/data/COVID/models/reg-adaptive-imaging'

fulldirs = [reg_no_imaging, reg_imaging_bloods]

for fulldir in fulldirs:
    # CSVs
    csvs = [f for f in pathlib.Path(fulldir).rglob('*.csv')]
    # TODO: Plot regressed values alongside actual labels
    # Color code according to ITU vs non-ITU: Get functionality from confusion_matrix_plotter.py
    # Start by reading csvs
    somedir = os.path.basename(fulldir)
    # Find csv(s)
    for csv in csvs:
        experiments_output = pd.read_csv(csv)
        # Separate network outputs from labels
        filenames = experiments_output.Filename
        if len(filenames[0]) > 50:
            new_filenames = []
            for filename in filenames:
                new_filenames.append(os.path.basename(filename))
            filenames = new_filenames
            experiments_output.Filename = filenames
        df_labels = experiments_output.Inv_Time_To_Death
        df_preds = experiments_output.Pred
        experiments_output['ICU'] = -1.0
        # Want to be able to separate predictions between ITU and non-ITU
        # Need to read in original labels to know if ITU admitted subject or not
        OG = pd.read_csv('/data/COVID/Labels/cxr_news2_pseudonymised_filenames_latest_folds.csv')
        for filename in experiments_output.Filename:
            # print(filename, OG[OG.Filename == filename]['ICU admission'])
            if OG[OG.Filename == filename]['ICU admission'].isnull().tolist()[0] and \
                    OG[OG.Filename == filename]['ICU admission2'].isnull().tolist()[0]:
                experiments_output.loc[experiments_output.Filename == filename, 'ICU'] = 0.0
            else:
                experiments_output.loc[experiments_output.Filename == filename, 'ICU'] = 1.0
        ICU_df = experiments_output[experiments_output['ICU'] == 1.0]
        non_ICU_df = experiments_output[experiments_output['ICU'] == 0.0]
        combined_dfs = [ICU_df, non_ICU_df]
        combined_titles = ['ICU_df', 'non_ICU_df']
        combined_colors = ['turquoise', 'darkorange']
        for df_num, dataframe in enumerate(combined_dfs):
            current_labels = dataframe.Inv_Time_To_Death
            current_preds = dataframe.Pred
            temp_labels = []
            temp_preds = []
            for i in range(len(current_labels)):
                temp_labels.append(eval(str(current_labels.iloc[i])))
                temp_preds.append(eval(str(current_preds.iloc[i])))
            temp_labels = np.array(temp_labels)
            sorted_temp_labels = np.array(sorted(temp_labels))
            temp_preds = np.array(temp_preds)
            sorted_temp_preds = np.array([x for _, x in sorted(zip(temp_labels, temp_preds))])
            # Overall MSE
            overall_mse = np.round(np.mean((sorted_temp_labels-sorted_temp_preds)**2), 4)
            plt.figure(np.random.randint(1000))
            plt.scatter(range(len(sorted_temp_labels)), sorted_temp_labels, color='black',
                        label=': '.join([combined_titles[df_num], 'Label']))
            plt.scatter(range(len(sorted_temp_labels)), sorted_temp_preds, color=combined_colors[df_num],
                        label=': '.join([combined_titles[df_num], 'Prediction', str(overall_mse) + ' MSE']))
            # plt.axis('square')
            plt.gca().set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
            plt.xlabel('Subjects', fontsize=16)
            plt.ylabel('Inverse time to death (1/days)', fontsize=16)
            plt.legend(fontsize=16)
            plt.title(f'Regression results for {somedir}', fontsize=18)
plt.show()


