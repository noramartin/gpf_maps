import numpy as np 
import networkx as nx 
from copy import deepcopy
import itertools
import matplotlib
import matplotlib.pyplot as plt
from os.path import isfile
import networkx as nx
from functools import partial
from scipy.stats import gmean
import pandas as pd
from matplotlib.lines import Line2D
from scipy.special import binom, comb
import seaborn as sns
import parameters as param
import parameters_plots as plotparam

####################################################################################################################################################
print('plot navigability vs # phenos and evolvability for all maps')
####################################################################################################################################################
ylims, xlims = [None, None], [None, None]
f, ax = plt.subplots(figsize=(5, 4))
for K in range(2, 6):
	for L in range(5, 14):
		navigability_filename = './data/fibonacci'+'_navigability_data_deleted_phenotypes_L'+str(L) +'_'+str(K)+'_iterations'+str(param.iterations_nav)+'_'+str(param.number_source_target_pairs)+ 'missing_phenos.csv'
		if isfile(navigability_filename):
			df = pd.read_csv(navigability_filename)
			if len(df['evolvability'].tolist()) == 0:
				print('no data', K, L)
				continue
			sc = ax.scatter(df['evolvability'].tolist(), df['number of phenotypes'].tolist(), c=df['navigability'].tolist(), vmin=0, vmax=1)
			if not ylims[0] or min(df['number of phenotypes'].tolist()) < ylims[0]:
				ylims[0] = min(df['number of phenotypes'].tolist())
			if not ylims[1] or max(df['number of phenotypes'].tolist()) > ylims[1]:
				ylims[1] = max(df['number of phenotypes'].tolist())		
			if not xlims[0] or min(df['evolvability'].tolist()) < xlims[0]:
				xlims[0] = min(df['evolvability'].tolist())
			if not xlims[1] or max(df['evolvability'].tolist()) > xlims[1]:
				xlims[1] = max(df['evolvability'].tolist())			
		else:
			print('not found', navigability_filename)
cb = f.colorbar(sc, ax=ax)
cb.set_label('navigability')
ax.set_xlabel(r'geometric mean of the NC evolvability $\bar{\epsilon}_{NC}$')
ax.set_ylabel('number of phenotypes')
####
n_data = np.power(10,np.linspace(np.log10(2), np.log10(ylims[1]*10), 10**4))
e_pred = [((n/10)**(1/n) - 1) * n for n in n_data]
ax.plot(e_pred, n_data, c='k')

####
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(ylims[0] - 1, 10 * ylims[1])
ax.set_xlim(max(0.5, xlims[0] - 2), 10 * xlims[1])
f.tight_layout()
f.savefig('./plots/navigability_evolv'+'_iterations'+str(param.iterations_nav) +'_' +str(param.number_source_target_pairs) +'.png', bbox_inches='tight', dpi=300)

####################################################################################################################################################
print('plot number of paths')
####################################################################################################################################################
L, K = 7, 3
max_path_length_to_compute = 100
ymin = 1/(param.iterations_nav * param.number_source_target_pairs)
f, ax = plt.subplots(ncols=2, figsize=(9, 4))#, sharex=True, sharey=True)
for i, type_model in enumerate(['standard', 'low_evolvability']):
	try:
		df_pl = pd.read_csv('./data/fibonaccinetwork_phenotypes_L'+str(L) +'_'+str(K)+'_iterations'+str(param.iterations_nav)+'_'+str(param.number_source_target_pairs)+type_model+ 'paths'+str(max_path_length_to_compute)+'.csv')
		phenotype_network_str_nodes = nx.read_gml('./data/fibonacciGPmap_L'+str(L)+'_'+str(K) +type_model+ '.gml')
		phenotype_network = nx.relabel_nodes(phenotype_network_str_nodes, mapping={n: int(n) for n in phenotype_network_str_nodes.nodes()}, copy=True)
		phenotpes_list = [ph for ph in phenotype_network.nodes()]	
		###
		#sns.lineplot(data=df_pl, x='path length', y='number paths', color = 'grey', estimator='mean', errorbar=('ci', 95), ax = ax[i])
		pl_list = sorted(list(set(df_pl['path length'].tolist())))
		path_length_vs_no_paths = {p: df_pl['number paths'][df_pl['path length'] == p].tolist() for p in pl_list}
		mean, std = [np.mean(path_length_vs_no_paths[p]) for p in pl_list], [np.std(path_length_vs_no_paths[p]) for p in pl_list]
		ax[i].errorbar(pl_list, mean, yerr=std,  c='grey', elinewidth=0.5)
		###
		nph = len(phenotpes_list)
		evolv = [len(set([ph2 for ph2 in phenotype_network.neighbors(ph)])) for ph in phenotpes_list]
		assert min(evolv) > 0 or type_model == 'low_evolvability'
		typical_evolv = gmean([e if e > 0 else 0.01 for e in evolv])
		analytic = [binom((nph - 2), l-1) * (typical_evolv/(nph-1))**l/l for l in range(1, max_path_length_to_compute + 1)]
		ax[i].plot(np.arange(1, max_path_length_to_compute + 1), analytic)
		ax[i].set_yscale('log')
		
		ax[i].set_ylim(ymin * 0.01, max(df_pl['number paths'].tolist()) * 2)
		ax[i].set_title({'standard': 'Fibonacci', 'low_evolvability': 'low-evolvability Fibonacci'}[type_model] + 
						 '\n' + r'$\bar{ \epsilon}_{NC} = $'+ str(round(typical_evolv)) + r', $n_p=$'+str(nph))
		ax[i].set_xlabel(r'path length $k$')
		ax[i].set_ylabel('number of\naccessible paths')
	except IOError:
		print('file not read')
		continue
for i in range(2):
	ax[i].set_xlim(0, 40)
f.tight_layout()
f.savefig('./plots/paths_L'+str(L)+'_'+str(K) +'_iterations'+str(param.iterations_nav)+'_'+str(param.number_source_target_pairs)+ 'paths'+str(max_path_length_to_compute)+'.png', dpi=200, bbox_inches='tight')
####################################################################################################################################################
print('plot navigability vs # phenos and evolvability for purely synthetic')
####################################################################################################################################################
navigability_filename = './data/sythetic_network_navigability_data_iterations' + \
    str(param.iterations_nav) + '_' + str(param.number_source_target_pairs) + '.csv'

if isfile(navigability_filename):
    f, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 10), sharex='col', sharey='col')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for rowindex, type_dist in enumerate(['constant', 'Poisson','geometric']):
        df = pd.read_csv(navigability_filename)
        sc = ax[rowindex, 2].scatter(
            df['evolvability'][df['type distribution'] == type_dist].tolist(),
            df['number of phenotypes'][df['type distribution'] == type_dist].tolist(),
            c=df['navigability'][df['type distribution'] == type_dist].tolist(), vmin=0, vmax=1)
        assert len(set(df['type distribution'][df['type distribution'] == type_dist].tolist())) == 1
        cb = f.colorbar(sc); cb.set_label('navigability')
        #########
        sc = ax[rowindex, 1].scatter(
            df['evolvability'][df['type distribution'] == type_dist].tolist(),
            df['number of phenotypes'][df['type distribution'] == type_dist].tolist(),
            c=df['coefficient of variation'][df['type distribution'] == type_dist].tolist(), 
            vmin=min(df['coefficient of variation'].tolist()) , vmax=max(df['coefficient of variation'].tolist()), cmap='plasma' )
        cb = f.colorbar(sc); cb.set_label('coefficient of variation\ndegree distr.')
        #
        xlims = (min(df['evolvability'].tolist()), max(df['evolvability'].tolist()))
        ylims = (min(df['number of phenotypes'].tolist()), max(df['number of phenotypes'].tolist()))
        for colindex in range(1, 3):
        	n_data = np.power(10,np.linspace(np.log10(ylims[0] * 0.5), np.log10(ylims[1]*2), 10**4))
        	e_pred = [((n/10)**(1/n) - 1) * n for n in n_data]
        	if colindex in [2, 3]:
        		ax[rowindex, colindex].plot(e_pred, n_data, c='k')
        	ax[rowindex, colindex].set_xlabel('evolvability')
        	ax[rowindex, colindex].set_ylabel('number of phenotypes')
        	ax[rowindex, colindex].set_yscale('log')
        	ax[rowindex, colindex].set_xscale('log')
        	ax[rowindex, colindex].set_ylim(0.5 * ylims[0], 2 * ylims[1])
        	ax[rowindex, colindex].set_xlim(0.5 * xlims[0], 2 * xlims[1])
        #############
        combinations_with_deg_file = [(nph, reps) for nph in [50, 100, 200, 300, 500, 10**3] for reps in np.arange(20, step=2) if isfile('./data/deg_sequence_'+str(type_dist)+'_'+str(nph)+'reps'+str(reps)+'.csv')]
        if len(combinations_with_deg_file):
            nph, reps = combinations_with_deg_file[-1]
            df_degree = pd.read_csv('./data/deg_sequence_'+str(type_dist)+'_'+str(nph)+'reps'+str(reps)+'.csv')
            ax[rowindex, 0].hist(df_degree['seq'], bins=np.arange(np.min(df_degree['seq']) - 0.5, np.max(df_degree['seq']) + 1.4))            
            ax[rowindex, 0].set_yscale('log')
            ax[rowindex, 0].set_xlabel('degree')
            ax[rowindex, 0].set_ylabel('frequency')#;ax[i, j].set_xlim(0, 800)
        ###############
        ax[rowindex, 1].set_title(type_dist)
    f.tight_layout()
    f.savefig('./plots/sythetic_network_navigability_data_iterations' +
              '_iterations' + str(param.iterations_nav) + '_' + str(param.number_source_target_pairs) +'.png', bbox_inches='tight', dpi=200)

