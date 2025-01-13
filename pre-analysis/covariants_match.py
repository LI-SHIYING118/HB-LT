#covariants matching between case and control group before epistasis analysis
#change the file name to your covariants file name and the column name to represent the covariants in your current file
#the file should be in a N*P+1, where N represent the number of inviduals and the P represent the number of covarinats, and one extra column indicate individual's assignment (1:0)
#if the after-mathcing file will be used for following selection, ID column will also be needed

def perform_matching(covariants_file):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    covariants_phenotype = pd.read_table(covariants_file)

    covariants = covariants_phenotype[['id', 'age_diag', 'diab_year', 'sex']]
    covariants = pd.get_dummies(covariants, columns=['sex'])
    covariants.replace({True: 1, False: 0}, inplace=True)
    covariants.drop('sex_f', axis=1, inplace=True)

    cols_to_scale = ['age_diag', 'diab_year', 'sex_m']
    scaler = StandardScaler()
    covariants[cols_to_scale] = scaler.fit_transform(covariants[cols_to_scale])

    # Prepare phenotype
    phenotype = covariants_phenotype[['id', 'class']]

    # Perform matching
    biobank_full = pd.merge(covariants, phenotype, on='id')

    macro_case = biobank_full['class'] == 1
    macro_control = biobank_full['class'] == 0
    macro_case = covariants[macro_case]
    macro_control = covariants[macro_control]
    macro_case.reset_index(inplace=True)
    macro_control.reset_index(inplace=True)

    nn_control = NearestNeighbors(n_neighbors=1)
    nn_control.fit(macro_control[['age_diag', 'diab_year', 'sex_m']].values)

    outputlist = []
    for i, row in macro_case.iterrows():
        case_patient_features = row[['age_diag', 'diab_year', 'sex_m']].values.reshape(1, -1)
        distances, neighbor_indices = nn_control.kneighbors(case_patient_features)
        neighbors = [macro_control.loc[idx, 'id'] for idx in neighbor_indices[0]]
        distances = distances[0].tolist()
        outputlist.append([row['id'], neighbors[0], distances[0]])

    output_MA = pd.DataFrame(outputlist,
                             columns=['case_id', 'control_id', 'distance'])

    return output_MA

# UK biobank Macrovascular
covariants_file = 'biobank_covariants_MA.txt'
biobank_MA = perform_matching(covariants_file)
print(biobank_MA)

