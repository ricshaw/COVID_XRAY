import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict


def plot_hists(df, title):
    print(df.shape) 
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(2,2,1)
    ax.set_title('Died')
    df['Died'].hist()
    ax = fig.add_subplot(2,2,2)
    ax.set_title('Age')
    df['Age'].hist(bins=10)
    ax = fig.add_subplot(2,2,3)
    ax.set_title('Gender')
    df['Gender'].hist()
    #ax = fig.add_subplot(2,2,4)
    #ax.set_title('Examination')
    #df['Examination_Title'].hist()
    df.Gender = df.Gender.replace('Male',1)
    df.Gender = df.Gender.replace('Female',0) 
    print('Age', df['Age'].mean(), df['Age'].std())
    print('Gender', df['Gender'].mean(), df['Gender'].std(), df[df.Gender==1].shape[0]/df[df.Gender==0].shape[0])
    print('Died', df['Died'].mean(), df['Died'].std(), df[df.Died==1].shape[0]/df[df.Died==0].shape[0])

def stratified_group_k_fold(X, y, groups, k, seed=None):
    """ https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in tqdm(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])), total=len(groups_and_y_counts)):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices



#file = 'cxr_news2_pseudonymised_filenames_jpgs_edit'
file = 'cxr_news2_pseudonymised_filenames_latest'
df = pd.read_csv(file + '.csv')

train_df, val_df = train_test_split(df, stratify=df.Died, test_size=0.10)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

plot_hists(train_df, 'Train')
plot_hists(val_df, 'Valid')
plt.show()

#train_df.to_csv('train.csv', index=False)
#val_df.to_csv('valid.csv', index=False)




df_folds = df
patient_id_2_count = df_folds[['patient_pseudo_id', 'Filename']].groupby('patient_pseudo_id').count()['Filename'].to_dict()
df_folds = df_folds.set_index('Filename', drop=False)
print(patient_id_2_count)


def get_stratify_group(row):
    #print(row['Age'])
    stratify_group = row['Gender']
    #stratify_group += f'_{row["Age"]}'
    stratify_group += f'_{row["Died"]}'
    patient_id_count = patient_id_2_count[row["patient_pseudo_id"]]
    #print(row['Age'], row['Gender'], row['Died'], patient_id_count)
    '''
    if patient_id_count > 80:
        stratify_group += f'_80'
    elif patient_id_count > 60:
        stratify_group += f'_60'
    elif patient_id_count > 50:
        stratify_group += f'_50'
    elif patient_id_count > 30:
        stratify_group += f'_30'
    elif patient_id_count > 20:
        stratify_group += f'_20'
    elif patient_id_count > 10:
        stratify_group += f'_10'
    else:
        stratify_group += f'_0'
    '''
    #print(stratify_group)
    return stratify_group


df_folds['stratify_group'] = df_folds.apply(get_stratify_group, axis=1)
df_folds['stratify_group'] = df_folds['stratify_group'].astype('category').cat.codes

df_folds.loc[:, 'fold'] = 0

skf = stratified_group_k_fold(X=df_folds.index, y=df_folds['stratify_group'], groups=df_folds['patient_pseudo_id'], k=5, seed=42)

for fold_number, (train_index, val_index) in enumerate(skf):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

print(set(df_folds[df_folds['fold']==0]['patient_pseudo_id'].values).intersection(df_folds[df_folds['fold']==1]['patient_pseudo_id'].values))
print(set(df_folds[df_folds['fold']==0]['patient_pseudo_id'].values).intersection(df_folds[df_folds['fold']==2]['patient_pseudo_id'].values))
print(set(df_folds[df_folds['fold']==0]['patient_pseudo_id'].values).intersection(df_folds[df_folds['fold']==3]['patient_pseudo_id'].values))
print(set(df_folds[df_folds['fold']==0]['patient_pseudo_id'].values).intersection(df_folds[df_folds['fold']==4]['patient_pseudo_id'].values))

df_folds.to_csv(file + '_folds.csv', index=False)
print(df_folds.head())
print('Final', df_folds.shape)

plot_hists(df_folds[df_folds['fold']==0], 'Fold 0')
plot_hists(df_folds[df_folds['fold']==1], 'Fold 1')
plot_hists(df_folds[df_folds['fold']==2], 'Fold 2')
plot_hists(df_folds[df_folds['fold']==3], 'Fold 3')
plot_hists(df_folds[df_folds['fold']==4], 'Fold 4')
plt.show()

