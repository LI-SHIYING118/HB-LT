#Extract patients based on matching results and filling the missing genotype by beagle

def stack_and_duplicate_ids(df, output_file):
    import pandas as pd
    df['case_id'] = df['case_id'].astype(int)
    df['control_id'] = df['control_id'].astype(int)
    result = pd.concat([df['case_id'], df['control_id']], ignore_index=True)
    result = pd.DataFrame(result)
    result[1] = result[0]
    result.to_csv(output_file, sep='\t', index=False, header=False)

#UK biobank
df = biobank_MA
output_file = 'biobank_MA_closest.txt'
stack_and_duplicate_ids(df, output_file)

#convert plink files into vcf files, for now, we only focus on chromosome 1
import subprocess

for chromosome in range(1, 2):
    plink_filename = f"ukb_imp_chr{chromosome}_17574_patients_diabetiques_UKbiobank_caucasiens_age_diag_sup_equ_30_ans_bed_from_bgen_ref_first_QC_data_avril"
    vcf_filename = f"ukb_imp_chr{chromosome}_17574_patients_diabetiques_UKbiobank_caucasiens_age_diag_sup_equ_30_ans_bed_from_bgen_ref_first_QC_data_avril"

    plink_command = f"plink --bfile {plink_filename} --keep biobank_MA_closest.txt --snps-only just-acgt --recode vcf --out {vcf_filename}"

    process = subprocess.Popen(plink_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode == 0:
        print(f"Conversion successful for chromosome {chromosome}")
    else:
        print(f"Error converting chromosome {chromosome}:")
        print(error.decode())

for chromosome in range(1, 2):
    input_filename = f"ukb_imp_chr{chromosome}_17574_patients_diabetiques_UKbiobank_caucasiens_age_diag_sup_equ_30_ans_bed_from_bgen_ref_first_QC_data_avril.vcf"
    output_filename = f"ukb_imp_chr{chromosome}_17574_patients_diabetiques_UKbiobank_caucasiens_age_diag_sup_equ_30_ans_bed_from_bgen_ref_first_QC_data_avril_imputed.vcf"

    beagle_command = f"java -Xmx10G -jar beagle.22Jul22.46e.jar gt={input_filename} out={output_filename}"

    process = subprocess.Popen(beagle_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode == 0:
        print(f"Imputation successful for chromosome {chromosome}")
    else:
        print(f"Error during imputation for chromosome {chromosome}:")
        print(error.decode())

import gc
gc.collect()

def data_sorting(vcf_file):
    import numpy as np
    import pandas as pd
    import allel
    covariants = pd.read_table(txt_file)
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

    bio_MA_id = geno_vcf['samples']
    bio_MA_id = pd.DataFrame(bio_MA_id)
    bio_MA_id[0] = bio_MA_id[0].str.split('_').str[0]
    bio_MA_id = bio_MA_id.rename(columns={0: 'id'})
    covariants['id'] = covariants['id'].astype('int64')
    bio_MA_id['id'] = bio_MA_id['id'].astype('int64')
    labels = pd.merge(bio_MA_id, covariants, left_on='id', right_on='id')
    labels['class']=labels['class'].astype(np.int_)
    print(labels)

    labels = list(labels['class'])


    return transposed_dosage, headers, labels

vcf_file='hapgen2_chr20_allwhite_1000.vcf'
transposed_dosage, headers, labels=data_sorting(vcf_file, txt_file)
df = pd.DataFrame(transposed_dosage, columns=headers)
print(df)
print(labels)

label=pd.DataFrame(labels)
label.to_csv('labels.csv')
#read the blocks file
import pandas as pd
block=pd.read_table('ukb_imp_chr1_17574_patients_diabetiques_UKbiobank_caucasiens_age_diag_sup_equ_30_ans_bed_from_bgen_ref_first_QC_data_avril_blocks.blocks', header=None)
print(block)



haplotype_blocks = []
for index, row in block.iterrows():
    snps = row[0].split()
    snps = [snp for snp in snps if snp.startswith('rs')]
    haplotype_blocks.append(snps)


for patient_id, patient_data in df.iterrows():
    haplotype_strings = []
    for snps in haplotype_blocks:
        haplotype_dosage = patient_data[snps]
        haplotype_string = ''.join(haplotype_dosage.astype(str))
        haplotype_strings.append(haplotype_string)


    concatenated_haplotypes = ' '.join(haplotype_strings)


    with open(f'patient_{patient_id}_haplotypes.txt', 'w') as file:
        file.write(concatenated_haplotypes + '\n')





def generate_haplotype_dataframe(df, block):
    haplotype_df = pd.DataFrame()

    for index, row in block.iterrows():
        snp_id_string = row.iloc[0].split()[1:]
        snp_dosages = df[snp_id_string].values.flatten()
        haplotype_genotype = ''.join(map(str, snp_dosages))
        haplotype_df = pd.concat([haplotype_df, pd.DataFrame([haplotype_genotype])])

    haplotype_df.reset_index(drop=True, inplace=True)
    return haplotype_df

result_df = generate_haplotype_dataframe(df, block)
print(result_df)
result_df.to_csv('result_df.txt')




import pandas as pd
pca=pd.read_csv('plink.dist', header=None, sep='\t')
print(pca)

#generate genetic similarity matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.close()
plt.figure(figsize=(10,10))
sns.heatmap(pca, cmap='viridis')
plt.savefig('F3.png')
