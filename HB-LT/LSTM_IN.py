import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HaplotypeLSTM:
    def __init__(self, phenotype_path, haplotype_dir='.', haplotype_prefix='output_',
                 test_size=0.4, random_state=42, max_blocks=200):
        """
        Haplotype block pre-filtering with LSTM

        Args:
            phenotype_path (str): Path to phenotype file (No head, one column)
            haplotype_dir (str): Directory containing haplotype block files
            haplotype_prefix (str): Prefix for haplotype filenames
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            max_blocks (int): Maximum number of blocks to process
        """
        self.phenotype_data = pd.read_csv(phenotype_path, header=None)
        self.haplotype_dir = Path(haplotype_dir)
        self.haplotype_prefix = haplotype_prefix
        self.test_size = test_size
        self.random_state = random_state
        self.max_blocks = max_blocks
        self.haplotype_blocks = None
        self.results = []

    def load_haplotypes(self):
        """Load haplotype blocks"""
        n_samples = len(self.phenotype_data)
        self.haplotype_blocks = [[] for _ in range(self.max_blocks)]

        for i in range(n_samples):
            file_path = self.haplotype_dir / f'{self.haplotype_prefix}{i}.txt'
            try:
                with open(file_path, 'r') as f:
                    blocks = [line.strip() for line in f.readlines()[:self.max_blocks]]

                    if len(blocks) < self.max_blocks:
                        raise ValueError(f"File {file_path} has only {len(blocks)} blocks")

                    for idx, block in enumerate(blocks[:self.max_blocks]):
                        self.haplotype_blocks[idx].append([int(allele) for allele in block])

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                raise

        # Convert to numpy arrays and validate shapes
        for idx in range(self.max_blocks):
            try:
                self.haplotype_blocks[idx] = np.array(self.haplotype_blocks[idx])
                if len(self.haplotype_blocks[idx].shape) != 2:
                    raise ValueError(f"Block {idx} has inconsistent dimensions")
            except ValueError as e:
                logger.error(f"Validation failed for block {idx}: {str(e)}")
                raise

    def create_model(self, input_length):
        """LSTM model"""
        model = Sequential([
            Embedding(input_dim=3, output_dim=12, input_length=input_length),
            LSTM(12, return_sequences=True),
            GlobalAveragePooling1D(),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss=self.rmse_metric)
        return model

    @staticmethod
    def rmse_metric(y_true, y_pred):
        """Root of mean square error"""
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def evaluate_block(self, block_idx):
        """Process each block independetly"""
        X = self.haplotype_blocks[block_idx]
        y = self.phenotype_data.values
        input_length = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        model = self.create_model(input_length)
        model.fit(X_train, y_train, epochs=100, verbose=0)

        kf = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        mse_scores = []

        for train_idx, test_idx in kf.split(X_test):
            X_fold, y_fold = X_test[test_idx], y_test[test_idx]
            y_pred = model.predict(X_fold)
            mse_scores.append(np.sqrt(mean_squared_error(y_fold, y_pred)))

        return mse_scores

    def analyze(self):
        """Main analysis pipeline"""
        if not self.haplotype_blocks:
            self.load_haplotypes()

        for block_idx in range(self.max_blocks):
            try:
                logger.info(f"Processing block {block_idx + 1}/{self.max_blocks}")
                scores = self.evaluate_block(block_idx)
                self.results.append({
                    'block': block_idx,
                    'mean_mse': np.mean(scores),
                    'std_mse': np.std(scores)
                })
            except Exception as e:
                logger.error(f"Failed processing block {block_idx}: {str(e)}")
                raise

        return pd.DataFrame(self.results)

    def plot_mse(self, results):
        """Generate a line plot based on mean_mse and std_mse"""
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            results['block'],
            results['mean_mse'],
            yerr=results['std_mse'],
            fmt='o-',
            capsize=5,
            label='Mean MSE ± Std Dev'
        )
        plt.xticks(results['block'].astype(int))
        plt.xlabel('Block Index')
        plt.ylabel('MSE')
        plt.title('Mean MSE and Standard Deviation per Block')
        plt.legend()
        plt.grid(True)
        plt.show()

    def filter_and_save_blocks(self, results, threshold, output_dir='filtered_haplotypes'):
        """
        Filter blocks based on MSE threshold and save updated haplotype files.

        Args:
            results (pd.DataFrame): DataFrame containing block analysis results
            threshold (float): MSE threshold for filtering blocks
            output_dir (str): Directory to save filtered haplotype files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Identify blocks to keep (above the threshold)
        selected_blocks = results[results['mean_mse'] > threshold]['block'].tolist()

        # Save the list of selected block IDs
        with open(output_dir / 'selected_blocks.txt', 'w') as block_file:
            block_file.write("\n".join(map(str, selected_blocks)))

        # Save updated haplotype files for each individual
        for i in range(len(self.phenotype_data)):
            file_path = self.haplotype_dir / f'{self.haplotype_prefix}{i}.txt'
            output_path = output_dir / f'{self.haplotype_prefix}{i}.txt'

            with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
                for block_idx, line in enumerate(infile):
                    if block_idx in selected_blocks:
                        outfile.write(line)

        logger.info(f"Saved filtered haplotype files to {output_dir}")
        logger.info(f"Selected block IDs saved to {output_dir / 'selected_blocks.txt'}")
