import subprocess
import pandas as pd
import numpy as np
import allel

class VCFToHaplotype:
    def __init__(self, vcf_file, plink_path='plink', output_prefix='output'):
        """
        Initialize the VCF to haplotype converter.

        :param vcf_file: Path to the VCF file.
        :param plink_path: Path to the PLINK executable (default: 'plink').
        :param output_prefix: Prefix for output files (default: 'output').
        """
        self.vcf_file = vcf_file
        self.plink_path = plink_path
        self.output_prefix = output_prefix

    def run_plink(self):
        """
        Run PLINK to generate haplotype blocks.
        """
        plink_command = f'{self.plink_path} --vcf {self.vcf_file} --blocks no-pheno-req --out {self.output_prefix}_block'
        process = subprocess.Popen(plink_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            print(f"PLINK error: {error.decode('utf-8')}")
        else:
            print("PLINK blocks generated successfully.")

    def load_vcf(self):
        """
        Load and process the VCF file.

        :return: Converting into a open dataframe.
        """
        types = {'calldata/GT': 'i2'}
        geno_vcf = allel.read_vcf(self.vcf_file, fields=['variants/ID', 'calldata/GT', 'samples'], types=types)

        print('Number of samples:', geno_vcf['samples'].shape[0])
        print('Sample IDs:', geno_vcf['samples'])
        print('Variant IDs:', geno_vcf['variants/ID'])
        print('Dosage:', geno_vcf['calldata/GT'])

        ids = geno_vcf['variants/ID']
        gt = geno_vcf['calldata/GT']
        imputed_dosage = gt[:, :, 0] + gt[:, :, 1]
        imputed_dosage = imputed_dosage.astype(np.uint8)
        transposed_dosage = imputed_dosage.T

        headers = list(geno_vcf['variants/ID'])
        df = pd.DataFrame(transposed_dosage, columns=headers)
        return df

    def load_blocks(self):
        """
        Load haplotype blocks from the PLINK output.

        :return: List of haplotype blocks.
        """
        block_file = f'{self.output_prefix}_block.blocks'
        block = pd.read_table(block_file, header=None)
        haplotype_blocks = []
        for index, row in block.iterrows():
            snps = row[0].split()
            snps = [snp for snp in snps if snp.startswith('rs')]
            haplotype_blocks.append(snps)
        return haplotype_blocks

    def generate_haplotype_files(self, df, haplotype_blocks):
        """
        Generate haplotype block files for each individual.

        :param df: DataFrame containing genotype dosages.
        :param haplotype_blocks: List of haplotype blocks.
        """
        for patient_id, patient_data in df.iterrows():
            haplotype_strings = []
            for snps in haplotype_blocks:
                haplotype_dosage = patient_data[snps]
                haplotype_string = ''.join(haplotype_dosage.astype(str))
                haplotype_strings.append(haplotype_string)

            concatenated_haplotypes = '\n'.join(haplotype_strings)

            with open(f'{self.output_prefix}_{patient_id}.txt', 'w') as file:
                file.write(concatenated_haplotypes + '\n')
        print(f"Haplotype files generated with prefix: {self.output_prefix}")

# Script-specific logic (runs only if you use this script individually)
if __name__ == "__main__":
    # Example usage
    vcf_file = 'your_file.vcf'
    converter = VCFToHaplotype(vcf_file, output_prefix='my_data')

    # Step 1: Run PLINK to generate blocks
    converter.run_plink()

    # Step 2: Load and process the VCF file
    df = converter.load_vcf()

    # Step 3: Load haplotype blocks
    haplotype_blocks = converter.load_blocks()

    # Step 4: Generate haplotype files for each individual
    converter.generate_haplotype_files(df, haplotype_blocks)