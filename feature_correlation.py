import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in the labels (unique subjects)
labels = pd.read_csv('/data/COVID/Labels/CXRs_latest_440_iterative.csv')
temp_bloods = labels[labels.columns.difference(labels.filter(like='ICU').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='date of death').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='OHE').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='stratify').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='fold').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='Death').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='Died').columns, sort=False)]
temp_bloods = temp_bloods.select_dtypes(include=[np.number])

num_features = len(temp_bloods.columns)
print(f'The number of features under scrutiny is {num_features}')

# Obtain all numeric columns (i.e.: bloods + age + comorbidities)
features_list = temp_bloods.columns.to_list()
died_array = np.array(labels['Died'])
correlations = []
p_values = []
for feature in features_list:
    feature_array = np.array(temp_bloods[feature])
    correlation, p_value = pearsonr(died_array, feature_array)
    if np.isnan(correlation):
        correlation = 0
    if np.isnan(p_value):
        p_value = 1
    correlations.append(correlation)
    p_values.append(p_value)

mvp_index = np.argmax(np.abs(correlations))
mvp_feature = temp_bloods.columns[mvp_index]
print(mvp_feature, correlations[mvp_index], p_values[mvp_index], mvp_index)

# Sort features according to correlation
sorted_features = [x for _, x in sorted(zip(np.abs(correlations), features_list))]
sorted_correlations = [x for _, x in sorted(zip(np.abs(correlations), correlations))]
sorted_p_values = [x for _, x in sorted(zip(np.abs(correlations), p_values))]

statistical_cutoff = np.argmax((np.array(sorted_p_values) < 0.05).astype(int))
# Plot histogram of correlations
# plt.figure()
# plt.bar(list(range(num_features)), sorted_correlations, tick_label=features_list)

# Plot
fig = plt.figure(figsize=(12, 8))
freq_series = pd.Series(sorted_correlations)
ax = freq_series.plot(kind='bar')
ax.set_title('Pearson correlation between each feature and death prediction')
# ax.set_xlabel('Amount ($)')
ax.set_ylabel('Pearson correlation')
ax.set_xticklabels(sorted_features)

rects = ax.patches
for rect, label in zip(rects, sorted_features):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+20, label,
            ha='center', va='bottom')
plt.gcf().subplots_adjust(bottom=0.3, top=0.9)  # adjusting the plotting area
plt.show()
