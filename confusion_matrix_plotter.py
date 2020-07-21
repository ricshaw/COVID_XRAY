import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import seaborn as sn
import os
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')

outputs_saved = True
plot_AUCs = False

num_classes = 2
if outputs_saved:
            #experiment = 'preds-efficientnet-b3-bs32-512-tta-ranger.csv'
            experiment = 'ensemble.csv'
            experiments_output_file = os.path.join('/nfs/home/richard', experiment)
            experiments_output = pd.read_csv(experiments_output_file)

            # Separate network outputs from labels
            filenames = experiments_output.Filename
            df_labels = np.stack((experiments_output.Died, 1-experiments_output.Died), axis=1)
            df_preds = np.stack((experiments_output.Pred, 1-experiments_output.Pred), axis=1)
            print(df_labels.shape, df_preds.shape)
            print(df_labels)
            print(df_preds)

            # Confusion matrix calcs
            # Convert OHE Labels, outputs
            standard_labels = []
            standard_preds = []

            for i in range(len(df_labels)):
                standard_labels.append(np.argmax(df_labels[i]))
                standard_preds.append(np.argmax(df_preds[i]))

            # AUC calcs
            # Convert OHE Labels, outputs
            labels = []
            preds = []

            for i in range(len(df_labels)):
                labels.append(df_labels[i])
                preds.append(df_preds[i])

            labels = np.array(labels)
            preds = np.array(preds)

            # Calculate confusion matrix
            class_names = ['Died', 'Survived']
            #class_names = ['48H', '1 week -', '1 week +', 'Survived', 'micro']
            conf_mat = confusion_matrix(standard_labels, standard_preds)
            l = len(conf_mat[0])
            diags = [conf_mat[i][i] for i in range(l)]
            conf_mat = conf_mat / np.array(diags)[None, ...].T
            index = [i+'_lab' for i in class_names]
            columns = [i for i in class_names]
            print(index)
            print(columns)
            df_cm = pd.DataFrame(conf_mat, index=index, columns=columns)
            plt.figure(figsize=(10, 7))
            plt.title('Confusion matrix for experiment: ' + experiment)
            ax = sn.heatmap(df_cm, annot=True)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            plt.savefig('confusion.png', dpi=300)
            exit(0)

            # Also plot AUCs
            if plot_AUCs:
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for classID in range(num_classes):
                    fpr[classID], tpr[classID], _ = roc_curve(labels[:, classID], preds[:, classID])
                    roc_auc[classID] = auc(fpr[classID], tpr[classID])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), preds.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # Compute PR curve and PR area for each class
                precision_tot = dict()
                recall_tot = dict()
                pr_auc = dict()
                for classID in range(num_classes):
                    precision_tot[classID], recall_tot[classID], _ = precision_recall_curve(labels[:, classID],
                                                                                            preds[:, classID])
                    pr_auc[classID] = auc(recall_tot[classID], precision_tot[classID])

                # Compute micro-average precision-recall curve and PR area
                precision_tot["micro"], recall_tot["micro"], _ = precision_recall_curve(labels.ravel(), preds.ravel())
                pr_auc["micro"] = auc(recall_tot["micro"], precision_tot["micro"])
                no_skill = len(labels[labels == 1]) / len(labels)

                colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'red']
                # Plot ROC-AUC for different classes:
                plt.figure()
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

                plt.figure()
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

else:
    # Need to load model and get outputs
    print('Not supported right now')
plt.show()
