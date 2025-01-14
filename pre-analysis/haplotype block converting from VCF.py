#the script will take the VCF file and convert it into a list of txt files.
#Each txt file represent a single individual's selected haplotype block profile
#The haplotype block is calculated by --blocks in PLINK

#If VCF files contain too many missing values, researchers can choice to using imputation methodes such as Beagle to
#fill the missing values before running this script

import subprocess
import pandas as pd

plink_command = f'plink --vcf file_name --blocks no-pheno-req --out file_name_block'
process = subprocess.Popen(plink_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, error = process.communicate()
print(error)


#load the VCF file
def data_sorting(vcf_file):
    import numpy as np
    import allel
    import pandas as pd
    types = {
        'calldata/GT': 'i2'}
    geno_vcf = allel.read_vcf(vcf_file, fields=['variants/ID', 'calldata/GT', 'samples'], types=types)

    print('Number of samples:', geno_vcf['samples'].shape[0])
    print('Sample IDs:', geno_vcf['samples'])
    print('Variant IDs:', geno_vcf['variants/ID'])
    print('Dosage:', geno_vcf['calldata/GT'])

    ids = geno_vcf['variants/ID']
    gt = geno_vcf['calldata/GT']
    imputed_dosage = gt[:, :, 0] + gt[:, :, 1]
    imputed_dosage = imputed_dosage.astype(np.uint8)
    transposed_dosage = imputed_dosage.T
    print(transposed_dosage)

    headers = list(geno_vcf['variants/ID'])
    print(headers)

    df = pd.DataFrame(transposed_dosage, columns=headers)

    return df
    #return transposed_dosage, headers

#real datasets
vcf_file='file_name.vcf'
df=data_sorting(vcf_file)


#Generating the haplotype block file for each individual
block=pd.read_table('file_name_BLOCKS.blocks', header=None)
print(block)


haplotype_blocks = []
for index, row in block.iterrows():
    snps = row[0].split()
    snps = [snp for snp in snps if snp.startswith('rs')]
    haplotype_blocks.append(snps)

print(haplotype_blocks)

for patient_id, patient_data in df.iterrows():
    haplotype_strings = []
    for snps in haplotype_blocks:
        haplotype_dosage = patient_data[snps]
        haplotype_string = ''.join(haplotype_dosage.astype(str))
        haplotype_strings.append(haplotype_string)

    concatenated_haplotypes = '\n'.join(haplotype_strings)

    with open(f'output_{patient_id}.txt', 'w') as file:
        file.write(concatenated_haplotypes + '\n')




