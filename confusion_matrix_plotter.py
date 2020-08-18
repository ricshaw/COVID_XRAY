import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import seaborn as sn
import os
import matplotlib.pyplot as plt
import matplotlib
import pathlib
from collections import Counter
matplotlib.use('TkAgg')


def AUC_plotter(labels, preds, class_names=None):
    # Compute ROC curve and ROC area for each class
    if type(labels) != list:
        num_classes = labels.shape[1]
        if num_classes > 2:
            if not class_names:
                class_names = ['48H', '1 week -', '1 week +', 'Survived', 'micro']
                class_names = class_names[:num_classes]
        else:
            if not class_names:
                class_names = ['Died']
    else:
        num_classes = len(labels)
        class_names.extend(['micro'])

    print(num_classes, class_names)
    # Compute ROC-AUCs
    if type(labels) != list:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for classID in range(num_classes):
            fpr[classID], tpr[classID], _ = roc_curve(labels[:, classID], preds[:, classID])
            roc_auc[classID] = auc(fpr[classID], tpr[classID])

        # Compute PR curve and PR area for each class
        precision_tot = dict()
        recall_tot = dict()
        pr_auc = dict()
        for classID in range(num_classes):
            precision_tot[classID], recall_tot[classID], _ = precision_recall_curve(labels[:, classID],
                                                                                    preds[:, classID])
            pr_auc[classID] = auc(recall_tot[classID], precision_tot[classID])
        if num_classes != 1:
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), preds.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute micro-average precision-recall curve and PR area
            precision_tot["micro"], recall_tot["micro"], _ = precision_recall_curve(labels.ravel(), preds.ravel())
            pr_auc["micro"] = auc(recall_tot["micro"], precision_tot["micro"])
        no_skill = len(labels[labels == 1]) / len(labels)
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for classID in range(num_classes):
            fpr[classID], tpr[classID], _ = roc_curve(labels[classID][:], preds[classID][:])
            roc_auc[classID] = auc(fpr[classID], tpr[classID])

        # Compute PR curve and PR area for each class
        precision_tot = dict()
        recall_tot = dict()
        pr_auc = dict()
        for classID in range(num_classes):
            precision_tot[classID], recall_tot[classID], _ = precision_recall_curve(labels[classID][:],
                                                                                    preds[classID][:])
            pr_auc[classID] = auc(recall_tot[classID], precision_tot[classID])
        if num_classes != 1:
            summed_labels = []
            summed_preds = []
            for exp in range(num_classes):
                summed_labels.extend(labels[exp])
                summed_preds.extend(preds[exp])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(summed_labels, summed_preds)
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute micro-average precision-recall curve and PR area
            precision_tot["micro"], recall_tot["micro"], _ = precision_recall_curve(summed_labels, summed_preds)
            pr_auc["micro"] = auc(recall_tot["micro"], precision_tot["micro"])

    colors = ['turquoise', 'darkorange', 'cornflowerblue', 'red', 'navy']
    # Plot ROC-AUC for different classes:
    plt.figure(np.random.randint(1000))
    plt.axis('square')
    for classID, key in enumerate(fpr.keys()):
        lw = 2
        plt.plot(fpr[key], tpr[key], color=colors[classID],  # 'darkorange',
                 lw=lw, label=f'{class_names[classID]} ROC curve (area = {roc_auc[key]: .2f})')
        plt.title(f'Class ROC-AUC for ALL classes: {somedir}', fontsize=18)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc="lower right")

    plt.figure(np.random.randint(1000))
    plt.axis('square')
    for classID, key in enumerate(precision_tot.keys()):
        lw = 2
        plt.plot(recall_tot[key], precision_tot[key], color=colors[classID],  # color='darkblue',
                 lw=lw, label=f'{class_names[classID]} PR curve (area = {pr_auc[key]: .2f})')
        plt.title(f'Class PR-AUC for ALL classes: {somedir}', fontsize=18)
        # plt.plot([0, 1], [0, 0], lw=lw, linestyle='--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.legend(loc="lower right")


outputs_saved = True
plot_conf = False
plot_AUCs = False
plot_ITUs = True
normalise_conf = True
plot_MVP = False

num_classes = 2
# fulldir = '/data/COVID/models/log-reg-all-features-ranger-iterative-imp-170/'
# fulldir = '/data/COVID/models/full-binary-cutmix-bloods-800-zero-impute-vitals'
# fulldir = '/data/COVID/models/full-binary-cutmix-bloods-80-zero-impute-vitals-occ0'
# fulldir = '/data/COVID/bootstrap/bootstrap_csvs'
# bloods_only_dir = '/data/COVID/models/full-binary-cutmix-bloods-100-zero-impute-vitals-occ0'
bloods_only_dir = '/data/COVID/models/bloods-only'
imaging_only_dir = '/data/COVID/models/imaging-only-death-480'
imaging_bloods_dir = '/data/COVID/models/imaging-bloods-death-480'

fulldirs = [bloods_only_dir, imaging_only_dir, imaging_bloods_dir]

for fulldir in fulldirs:
    # Find somedir
    somedir = os.path.basename(fulldir)
    # Find csv(s)
    csvs = [f for f in pathlib.Path(fulldir).rglob('*.csv')]

    # Some overall variables
    overall_feature_counter = Counter({})
    for csv in csvs:
        csv = str(csv)
        print(csv)
        if outputs_saved:
            # for subdir, dirs, files in os.walk('/data/COVID/models'):
            #     for somedir in dirs:
                    # Outputs saved from training (potentially from all folds)
                    # fulldir = os.path.join('/data/COVID/models', somedir)
            # csv = [f for f in pathlib.Path(fulldir).rglob('*.dcm')]
            # experiments_output_file = os.path.join(fulldir, 'preds.csv') # '/data/COVID/models/death-time-b3-folds-tta/preds-efficientnet-b3-bs32-512.csv'
            experiments_output = pd.read_csv(csv)

            # Separate network outputs from labels
            filenames = experiments_output.Filename
            df_labels = experiments_output.Died
            df_preds = experiments_output.Pred

            # Confusion matrix calcs
            # Convert OHE Labels, outputs
            standard_labels = []
            standard_preds = []

            if num_classes > 2:
                for i in range(len(df_labels)):
                    standard_labels.append(np.argmax(eval(str(df_labels[i]))))
                    standard_preds.append(np.argmax(eval(str(df_preds[i]))))
            else:
                if type(df_labels[0]) == list:
                    for i in range(len(df_labels)):
                        standard_labels.append(eval(str(df_labels[i]))[0])
                        standard_preds.append(eval(str(df_preds[i]))[0])
                    standard_preds = np.round(standard_preds)
                else:
                    for i in range(len(df_labels)):
                        standard_labels.append(eval(str(df_labels[i])))
                        standard_preds.append(eval(str(df_preds[i])))
                    standard_preds = np.round(standard_preds)
            # AUC calcs
            # Convert OHE Labels, outputs
            labels = []
            preds = []

            for i in range(len(df_labels)):
                labels.append(eval(str(df_labels[i])))
                preds.append(eval(str(df_preds[i])))

            labels = np.array(labels)
            preds = np.array(preds)

            if plot_conf:
                # Calculate confusion matrix
                conf_mat = confusion_matrix(standard_labels, standard_preds)

                # Normalise
                # conf_mat = (conf_mat.T / conf_mat.sum(axis=1)).T
                if normalise_conf:
                    l = len(conf_mat[0])
                    # off_diags = [conf_mat[l-1-i][i] for i in range(l-1, -1, -1)]
                    diags = [conf_mat[i][i] for i in range(l)]
                    conf_mat = conf_mat / np.array(diags)[None, ...].T

                df_cm = pd.DataFrame(conf_mat, index=[i+'_lab' for i in class_names[:-1]],
                                     columns=[i for i in class_names[:-1]])
                plt.figure(figsize=(10, 7))
                plt.title(f'Confusion matrix for experiment: {somedir}')
                sn.heatmap(df_cm, annot=True)

            if plot_MVP:
                df_preds = experiments_output.MVP_feat.to_list()
                feature_counts = Counter(df_preds)
                overall_feature_counter += feature_counts
                df = pd.DataFrame.from_dict(feature_counts, orient='index')
                df.columns = ['Features']
                df.sort_values('Features', inplace=True)
                from matplotlib import rcParams
                rcParams.update({'figure.autolayout': True})
                ax = df.plot(kind='bar')
                ax.set_title("Model feature importance: Counts")
                plt.savefig(csv[:-4] + '.png')

            # Also plot AUCs
            if plot_AUCs:
                AUC_plotter(labels, preds)

            if plot_ITUs:
                experiments_output['ICU'] = -1.0
                # Want to be able to separate predictions between ITU and non-ITU
                # Need to read in original labels to know if ITU admitted subject or not
                OG = pd.read_csv('/data/COVID/Labels/cxr_news2_pseudonymised_filenames_latest_folds.csv')
                for filename in experiments_output.Filename:
                    # print(filename, OG[OG.Filename == filename]['ICU admission'])
                    if OG[OG.Filename == filename]['ICU admission'].isnull().tolist()[0] and OG[OG.Filename == filename]['ICU admission2'].isnull().tolist()[0]:
                        experiments_output.loc[experiments_output.Filename == filename, 'ICU'] = 0.0
                    else:
                        experiments_output.loc[experiments_output.Filename == filename, 'ICU'] = 1.0
                ICU_df = experiments_output[experiments_output['ICU'] == 1.0]
                non_ICU_df = experiments_output[experiments_output['ICU'] == 0.0]
                combined_dfs = [ICU_df, non_ICU_df]
                combined_titles = ['ICU_df', 'non_ICU_df']
                combined_colors = ['turquoise', 'darkorange']
                # Plot AUCs now
                full_labels = []
                full_preds = []
                for df_num, dataframe in enumerate(combined_dfs):
                    current_labels = dataframe.Died
                    current_preds = dataframe.Pred
                    temp_labels = []
                    temp_preds = []
                    for i in range(len(current_labels)):
                        temp_labels.append(eval(str(current_labels.iloc[i])))
                        temp_preds.append(eval(str(current_preds.iloc[i])))
                    temp_labels = np.array(temp_labels)
                    temp_preds = np.array(temp_preds)
                    full_labels.append(temp_labels)
                    full_preds.append(temp_preds)
                AUC_plotter(full_labels, full_preds, combined_titles)

        else:
            # Need to load model and get outputs
            print('Not supported right now')
    if plot_MVP:
        overall_df = pd.DataFrame.from_dict(overall_feature_counter, orient='index')
        overall_df.columns = ['Features']
        overall_df.sort_values('Features', inplace=True)
        rcParams.update({'figure.autolayout': True})
        ax = df.plot(kind='bar')
        ax.set_title("Overall Model feature importance: Counts")
        plt.savefig('Overall feature counter' + '.png')
plt.show()
