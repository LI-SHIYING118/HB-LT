import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class PCA_filtering:
    def __init__(self, file_path, id_column='id'):
        self.file_path = file_path
        self.id_column = id_column
        self.covariants = None
        self.pca_df = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def load_data(self):
        """Load the data from your file."""
        self.covariants = pd.read_table(self.file_path)
        return self.covariants

    def preprocess_data(self):
        """ dropping NA values and standardizing."""
        self.covariants = self.covariants.dropna()
        self.covariants.reset_index(drop=True, inplace=True)
        columns_to_standardize = self.covariants.columns.difference([self.id_column])
        self.covariants[columns_to_standardize] = self.scaler.fit_transform(self.covariants[columns_to_standardize])
        return self.covariants

    def perform_pca(self):
        """Perform 2d PCA on the standardized data."""
        columns_to_standardize = self.covariants.columns.difference([self.id_column])
        principal_components = self.pca.fit_transform(self.covariants[columns_to_standardize])
        self.pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        self.pca_df[self.id_column] = self.covariants[self.id_column]
        return self.pca_df

    def visualize_pca(self, gridsize=50, cmap='inferno'):
        """Visualize the result."""
        plt.figure(figsize=(10, 7))
        hb = plt.hexbin(self.pca_df['PC1'], self.pca_df['PC2'], gridsize=gridsize, cmap=cmap)
        cb = plt.colorbar(hb, label='Counts')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()
        plt.show()

    def select_dense_region(self, x_start, x_end, y_start, y_end):
        """Select and visualize the most dense region in the PCA plot. Region will be decided by researchers themselves"""
        current_axis = plt.gca()
        rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                         linewidth=2, edgecolor='green', facecolor='none', linestyle='dashed')
        current_axis.add_patch(rect)
        plt.show()

        dense_ids = self.pca_df[
            (self.pca_df['PC1'] >= x_start) & (self.pca_df['PC1'] <= x_end) &
            (self.pca_df['PC2'] >= y_start) & (self.pca_df['PC2'] <= y_end)
        ][self.id_column].tolist()

        dense_ids = pd.DataFrame(dense_ids, columns=[self.id_column])
        return dense_ids
