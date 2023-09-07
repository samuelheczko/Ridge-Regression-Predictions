import pandas as pd
import connectome

path = '../data'

netmat = pd.read_csv(path + '/results/connectomes/HCP/netmats2.txt',sep=" ",header=None)
sub_ids = pd.read_csv(path + '/results/connectomes/HCP/subjectIDs.txt',sep=" ",header=None)
gaby_ids = pd.read_csv(path + '/results/connectomes/HCP/subj390.txt',sep=" ",header=None)

our_subs = netmat.iloc[(pd.concat([gaby_ids,sub_ids]).duplicated().iloc[gaby_ids.shape[0]:].values)].to_numpy()
our_subs = our_subs.reshape(390,100,100)

connectome.save_connectomes_df(correlation_matrices = our_subs,anatomical_label_presence = False,anatomic_labels  = None, path_to_save = path + '/results/connectomes/HCP/', atlas_name = 'HCP_ICA', n_subjects = 390, correlation_measure = 'pearson',subject_ixds = sub_ids.values.flatten())