import pandas as pd
import matplotlib.pyplot as plt
#Loading dataset
df = pd.read_csv('creditcard.csv')
#Description of data
print(df.info())
print(df.describe())
#Occurence 
occ = df['Class'].value_counts()
print(occ)
print(occ /len(df))
#PDF 
import matplotlib.pyplot as plt
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
num_rows = 10
num_cols = 3
total_plots = num_rows * num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))
axes = axes.flatten()
summary_stats=df.describe()
for i in range(len(x[0])):
    axes[i].hist(x[:, i]) 
    axes[i].set_title(f'Feature {i+1}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    stats_text = f"Mean: {summary_stats.iloc[1, i]:.2f}\nStd Dev: {summary_stats.iloc[2, i]:.2f}\nMin: {summary_stats.iloc[3, i]:.2f}\nMax: {summary_stats.iloc[7, i]:.2f}"
    axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8, color='black')


plt.tight_layout()
plt.show()
#Boxplots
num_rows = 10
num_cols = 3
total_plots = num_rows * num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))
axes = axes.flatten()
summary_stats=df.describe()
for i in range(len(x[0])):
    axes[i].boxplot(x[:, i]) 
    axes[i].set_title(f'Feature {i+1}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    stats_text = f"Mean: {summary_stats.iloc[1, i]:.2f}\nStd Dev: {summary_stats.iloc[2, i]:.2f}\nMin: {summary_stats.iloc[3, i]:.2f}\nMax: {summary_stats.iloc[7, i]:.2f}"
    axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8, color='black')


plt.tight_layout()
plt.show()
#Relation between features and the output 
num_rows = 10
num_cols = 3
total_plots = num_rows * num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))
axes = axes.flatten()
summary_stats=df.describe()
for i in range(len(x[0])):
    axes[i].scatter(x[:, i],y) 
    axes[i].set_title(f'Feature {i+1}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    stats_text = f"Mean: {summary_stats.iloc[1, i]:.2f}\nStd Dev: {summary_stats.iloc[2, i]:.2f}\nMin: {summary_stats.iloc[3, i]:.2f}\nMax: {summary_stats.iloc[7, i]:.2f}"
    axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8, color='black')


plt.tight_layout()
plt.show()



