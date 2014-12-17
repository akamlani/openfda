import optparse
from optparse import OptionParser

import time
from pprint import pprint
import numpy as np
import pandas as pd

import json
import openfda as ofda

import ml_alg as alg
from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

if __name__ == '__main__':
	# read in command line options
	parser = OptionParser()
	desc="OpenFDA Drug Reaction Prediction Classifier"
	parser = optparse.OptionParser(description=desc, usage="usage: %prog [options] filename", version="%prog version 1.0")
	parser.add_option("-f", "--file", action='store',  dest="api_type", help="api or file", default=1, type="int")
	
	(args, opts) = parser.parse_args()
	file_type = args.api_type
	print "API Type:{0}".format(file_type)
	# ipython openfda.py -- -f 0

	results = []
	if not file_type:
		results = ofda.load_from_api()
	else:
		results = ofda.load_from_file()

	# Shuffle the data and take training data as first 3000 samples
	results_in = shuffle(results, random_state=0)
	training_samples = results_in[:3000]

	drug_table, reaction_table, name_drug_v, le_l = ofda.prepare_data(training_samples)
	n_drugs = len(le_l[1].classes_)
	n_reactions = len(le_l[2].classes_)	
	# Being Neural Network RBM Training: Calculate Cosine Simalarity for how drug are related according to Mapped Weights
	start = time.time()
	mappings, dists, rbm_o = alg.rbm_dnn(name_drug_v.values, n_drugs, n_reactions, list(le_l[1].classes_) )	
	end = time.time()
	print "Training Time= {0}, {1}, {2}".format(end-start, start, end)
	

	# Training Sequence Complete - How to we Perform on New Data we havent seen yet
	# Feed new data into Network - See if it can generalize
	# IMPLEMENT LATER, for now just go off Training Sequence Data
	# TBD: This may be incorrect as it is possible the visible units are different from training and test sets
	# NOTE: This produces an alignment error due to wrong dimensions: visible unit dimensions must be the same!!!
	# test_samples = results[3000:5000]
	# drug_table, reaction_table, name_drug_v, le_l = ofda.prepare_data(test_samples)
	# n_drugs = len(le_l[1].classes_)
	# n_reactions = len(le_l[2].classes_)	
	# mappings, dists = alg.rbm_dnn_run(rbm_o, name_drug_v.values, n_drugs, n_reactions, list(le_l[1].classes_) )


	#### At this point we have trained the data and run new samples through it, the remaining code is analysist ####
	#### Analysis based on the weights of the hidden nodes ####

	# Create Drug/Reaction Matrix
	drug_reaction = pd.merge(reaction_table,drug_table, on='safetyreportid')
	drug_reaction = drug_reaction[['safetyreportid', 'reactionmeddrapt', 'medicinalproduct']]
	drug_reaction = drug_reaction.set_index(['safetyreportid', 'medicinalproduct'])
	drugs_l_r, react_l, dr_v = ofda.drug_reaction_binary_vec(drug_reaction)
	drug_react_v = pd.DataFrame(dr_v, index=drugs_l_r, columns=react_l)
	print "parse drug/reaction vector complete: {0}".format(drug_react_v.shape)

	# For each corresponding Hidden Node (Cluster), Find the Top 3 Most Drug Positively Weighted Correlations
	# Unlabeled Reaction Highly Correlated with these drugs
	labels = [item for item in dists.index]
	labels = list( set(labels) - set(['Bias Unit']) )
	hidden_nodes = [item for item in mappings.columns] 
	hidden_nodes = hidden_nodes[1:]
	n_top_drugs = 2
	hidden_names_i = []
	for h_node in hidden_nodes:
		df_hidden_col =  mappings.sort(columns=h_node, ascending=False)
		top_corr_drugs = df_hidden_col[:n_top_drugs].index
		top_drug_labels = [item for item in top_corr_drugs] 
		top_drug_labels = list( set(top_drug_labels) - set(['Bias Unit']) )
		top_drug_idices = le_l[1].transform(top_drug_labels)
		# correlate with original matrix and see if these drugs were taken together, record Patient IDs
		cond = name_drug_v[top_drug_idices].sum(axis=1) >= n_top_drugs
		names_v = [item for item in name_drug_v[cond].index]
		if names_v != []: 
			# for those drugs taken together, determine what reaction(s) occured
			drug_react_i = drug_react_v.loc[top_drug_idices]
			m_reaction = drug_react_i[drug_react_i[drug_react_v.columns] == 1].T
			m_reaction_l =  [(drug, list(m_reaction[drug].dropna().index)) for drug in top_drug_idices]  
			# for those drugs taken together, what reactions of this subset of patients occured
			for name in names_v:			
				report_id_m = reaction_table[reaction_table['safetyreportid'] == name]
				reactions_id_m = list(report_id_m['reactionmeddrapt'])
				hidden_names_i.append((h_node, top_drug_idices, name, reactions_id_m, m_reaction_l))
				
	
	# select top most occurences of Patient Drug Contributions
	# this is foundation from which we select other infomation critieria
	id_l = []
	for node in hidden_names_i: id_l.append(node[2])
	id_freq = pd.Series(id_l).value_counts()[:3]
	plt.bar(range(len(id_freq)), id_freq.values)
	plt.xticks(range(len(id_freq)), list(str(id) for id in id_freq.index), rotation=90)
	plt.title('Patient Frequency over Multiple Drugs')
	plt.savefig('../plots/' + 'top_patient_distribution' + '.png')    

	# of those drugs taken, and in hidden nodes, what top patients took these drugs
	l = []; r_intersect = []; mdl = []
	df = pd.DataFrame(hidden_names_i, columns=['hidden node', 'drugs', 'safetyreportid', 'id_reactions', 'drug_reactions'])
	top_idlist = list(str(id) for id in id_freq.index)
	for name, group in df.groupby('safetyreportid'):
		if name in top_idlist:
			d = []; t = list(group.drugs.values)
			for drug in t: 
				for v in drug: d.append(v)
			l.append([name, d])

			# find the intersection of reactions of drugs taken by patients and overall reactions across drugs taken
			r = []; id_reactions = group.id_reactions.values[0]
			for dr in group['drug_reactions']:
				for idx in dr: r.append([idx[0], list( set(idx[1]) & set(id_reactions) )] )
				
			r_intersect.append([name,r])


	names_l = [n[0] for n in l]
	# for patients vs taken drugs	
	for idx in l: 
		for v in idx[1]: mdl.append(v) 
	mdl = np.unique(np.sort(mdl))
	filtered_dg_v = name_drug_v.loc[names_l]
	filtered_dg_v = filtered_dg_v[mdl]

	# of those drugs taken, what are all possible reactions that occured
	drug_react_i = drug_react_v.loc[mdl]
	m_reaction = drug_react_i[drug_react_i[drug_react_v.columns] == 1].T
	m_reaction_l =  [(drug, list(m_reaction[drug].dropna().index)) for drug in mdl]  
	mrl = []
	for m in m_reaction_l:
		for v in m[1]: mrl.append(v)
	mrl = np.unique(np.sort(mrl))
	filtered_dr_v = drug_react_v.loc[mdl]
	filtered_dr_v = filtered_dr_v[mrl]

	# of those drugs taken, what is the intersection of all all reaction of drugs and patient reactions
	rdl = []; mrrdl = []
	for inter in r_intersect:
		for seq in inter[1]: 
			rdl.append(seq[0])
			for v in seq[1]: mrrdl.append(v)
	rdl = np.unique(np.sort(rdl))		
	mrrdl = np.unique(np.sort(mrrdl))
	dr_red_v = np.zeros( [len(rdl), len(mrrdl) ] )
	dr_red_v = pd.DataFrame(dr_red_v, index=rdl, columns=mrrdl)
	for inter in r_intersect:
		for seq in inter[1]:
			drug = seq[0]; react = seq[1]
			if set(react).issubset(set(mrrdl)): 
				r = list( set(react) & set(mrrdl) )
				dr_red_v.loc[drug][r] = 1
		


	# Look at Heatmap for Patient and Drugs Taken
	alg.plot_heat_map(filtered_dg_v, le_l[1].inverse_transform(mdl), names_l, 'Patient vs Drugs Taken', 'names_drugs')
	# Look at heat map plots of drugs vs reactions (overall)
	alg.plot_heat_map(filtered_dr_v, le_l[2].inverse_transform(mrl), le_l[1].inverse_transform(mdl), 'Drugs vs Reactions Taken', 'drug_reaction_all')
	# heat map plots from neural network (drugs taken (visible nodes) vs hidden Nodes)
	ann = mappings.loc[le_l[1].inverse_transform(mdl)]
	ann = ann[df['hidden node']]
	alg.plot_heat_map(ann, ann.columns, ann.index, 'Visible (Drug) vs Hidden (Mapped) States', 'drug_hidden_rbm')
	# heat map plots from neural network (cosine similarity - based on drugs taken by samples)
	drug_cos = dists.loc[le_l[1].inverse_transform(mdl)]
	drug_cos = drug_cos[le_l[1].inverse_transform(mdl)]
	alg.plot_heat_map(drug_cos, drug_cos.columns, drug_cos.index, 'Drug Correlation', 'drug_corr_rbm')
	# heat map plots for intersection of Patient Reaction and Overall Reactions correlated to drug
	dr_red_v.columns = le_l[2].inverse_transform(mrrdl)
	dr_red_v.index = le_l[1].inverse_transform(rdl)
	alg.plot_heat_map(dr_red_v, dr_red_v.columns, dr_red_v.index, 'Drug Reaction Minimization', 'drug_reaction_minimize')

	# overall top drugs with highest correlation with one another (based on output of ann)
	dists = dists.sort(columns=labels, ascending=False)
	top_corr_drugs = dists[:10].index
	top_drug_labels = [item for item in top_corr_drugs] 
	top_drug_labels = list( set(top_drug_labels) - set(['Bias Unit']) )
	top_drug_idices = le_l[1].transform(top_drug_labels)
	print "Top 10 Most Correlated Drug Samples: "
	pprint(top_drug_labels)



	# Run Clustering on drug information for further grouping
	# TBD





 










