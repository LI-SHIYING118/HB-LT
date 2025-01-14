import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#Pre-filtering individuals based on their covariants

covariants=pd.read_table('you_file_name.txt')
print(covariants)

#If your files contains some individuals with a lot of NA values, we recommond you drop them before continuing
covariants = covariants.dropna()
covariants.reset_index(drop=True, inplace=True)

# Selecting the columns to standardize
id_column = covariants['id']
columns_to_standardize = covariants.columns.difference(['id'])

scaler = StandardScaler()
covariants[columns_to_standardize] = scaler.fit_transform(covariants[columns_to_standardize])
print(covariants)

#converting the reults into a 2d plot by PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(covariants[columns_to_standardize])
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['id'] = id_column
print(pca_df)

#visulize your results
plt.figure(figsize=(10, 7))
hb = plt.hexbin(pca_df['PC1'], pca_df['PC2'], gridsize=50, cmap='inferno')
cb = plt.colorbar(hb, label='Counts')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

#select the most dense region in your plot
#the number will be decided by researchers themselves
x_start, x_end = 1, 0
y_start, y_end = 1, 0

# Add the square to the plot
current_axis = plt.gca()
rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                 linewidth=2, edgecolor='green', facecolor='none', linestyle='dashed')
current_axis.add_patch(rect)

plt.show()

#extract the IDs for selected individuals
dense_ids = pca_df[
    (pca_df['PC1'] >= 1) & (pca_df['PC1'] <= 0) &
    (pca_df['PC2'] >= 1) & (pca_df['PC2'] <= 0)
]['id'].tolist()

dense_ids=pd.DataFrame(dense_ids)
print(dense_ids)

dense_ids = dense_ids.rename(columns={0: 'id'})
print(dense_ids)
