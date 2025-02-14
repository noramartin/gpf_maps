import numpy as np 
import networkx as nx 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from os.path import isfile
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import parameters as param
import parameters_plots as plotparam
from functions.navigability_functions import isundefinedpheno, load_dict


####################################################################################################################################################
print('number of NCs for all models - info for table')
####################################################################################################################################################
map_vs_number_NCs, map_vs_number_pheno, map_vs_zero_evolv_NCs, map_vs_undefined, map_vs_mean_size_of_zero_evolv_NCs = {}, {}, {}, {}, {}
for type_structure in param.all_types_structue:
	NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
	ph_evolv_filename = './data/ph_vs_evolv'+type_structure+ '.csv'
	NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
	if isfile(NC_evolv_filename):
		NC_vs_evolv = load_dict(NC_evolv_filename)
		ph_vs_evolv = load_dict(ph_evolv_filename)
		NC_vs_size = load_dict(NC_vs_size_filename)
		map_vs_number_NCs[type_structure] = len([type_structure for NC, e in NC_vs_evolv.items() if NC > 0])
		map_vs_number_pheno[type_structure] = len([type_structure for NC, e in ph_vs_evolv.items() if NC > 0])
		map_vs_zero_evolv_NCs[type_structure] = len([e for NC, e in NC_vs_evolv.items() if NC > 0 and e == 0])
		if map_vs_zero_evolv_NCs[type_structure]:
			map_vs_mean_size_of_zero_evolv_NCs[type_structure] = int(round(np.mean([NC_vs_size[NC] for NC, e in NC_vs_evolv.items() if NC > 0 and e == 0])))
		else:
			map_vs_mean_size_of_zero_evolv_NCs[type_structure] = ''
		map_vs_undefined[type_structure] = 1 - sum([NC_vs_size[NC] for NC in NC_vs_evolv.keys() if NC > 0])/param.type_structure_vs_K[type_structure]**param.type_structure_vs_L[type_structure] 
for type_structure in map_vs_number_NCs:
	print(plotparam.type_structure_vs_label[type_structure], '&', map_vs_number_NCs[type_structure],  '&', param.type_structure_vs_K[type_structure]**param.type_structure_vs_L[type_structure], '&', map_vs_number_pheno[type_structure], '&', map_vs_zero_evolv_NCs[type_structure], '&', map_vs_mean_size_of_zero_evolv_NCs[type_structure], '&', int(round(100*map_vs_undefined[type_structure])), '\%', '\\', '\n', '\hline')

####################################################################################################################################################
print('checks against Greenbury data tables')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 5, figsize=(13.5, 2.8), gridspec_kw={'width_ratios': [1,]*4 + [0.5]})
for i, characteristic in enumerate(['navigability', 'number of NCs', 'number of phenotypes', 'fraction undefined']):
	for j, type_structure in enumerate(param.all_types_structue):
		if characteristic == 'navigability':
			nav_filename = './data/navigability'+type_structure+'_iterations'+str(param.iterations_nav)+'number_source_target_pairs'+str(param.number_source_target_pairs)+'.npy'
			if isfile(nav_filename)  and type_structure in plotparam.type_structure_vs_navigability:
				nav_list = np.load(nav_filename)
				ax[i].scatter([plotparam.type_structure_vs_navigability[type_structure],], [np.mean(nav_list),], c=plotparam.type_structure_vs_color[type_structure], s=15, marker='x')
		if characteristic == 'number of NCs':
			if type_structure in map_vs_number_NCs  and type_structure in plotparam.type_structure_vs_no_NCs:
				ax[i].scatter([plotparam.type_structure_vs_no_NCs[type_structure],], [map_vs_number_NCs[type_structure],], c=plotparam.type_structure_vs_color[type_structure], s=15, marker='x')
				assert plotparam.type_structure_vs_no_NCs[type_structure] == map_vs_number_NCs[type_structure]
		if characteristic == 'number of phenotypes':
			if type_structure in map_vs_number_pheno  and type_structure in plotparam.type_structure_vs_nph:
				ax[i].scatter([plotparam.type_structure_vs_nph[type_structure] - 1,], [map_vs_number_pheno[type_structure],], c=plotparam.type_structure_vs_color[type_structure], s=15, marker='x')
				assert plotparam.type_structure_vs_nph[type_structure] == map_vs_number_pheno[type_structure] + 1 # Greenbury data counts undefined
		if characteristic == 'fraction undefined':
			if type_structure in map_vs_undefined and type_structure in plotparam.type_structure_vs_fraction_unbound:
				ax[i].scatter([plotparam.type_structure_vs_fraction_unbound[type_structure],], [map_vs_undefined[type_structure],], c=plotparam.type_structure_vs_color[type_structure], s=15, marker='x')
		ax[i].set_xlabel(characteristic+' Greenbury')
		ax[i].set_ylabel(characteristic)
		if 'number' in characteristic:
			ax[i].set_xscale('log')
			ax[i].set_yscale('log')	
for i in range(4):
	ax[i].set_aspect('equal', 'box')	
	lims = ax[i].get_xlim()
	ax[i].plot(lims, lims, c='k', zorder=-5)	
ax[-1].legend(handles=plotparam.legend_without_Fibonaccinull)
ax[-1].axis('off')
f.tight_layout()
f.savefig('./plots/Greenbury_comparison_iterations'+str(param.iterations_nav)+'.pdf', bbox_inches='tight')

####################################################################################################################################################
print('size vs evolv in each GP map')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 5.5), sharex=True, sharey=True)
for i, type_structure in enumerate(param.all_types_structue):
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
	NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
	if isfile(NC_evolv_filename):
		NC_vs_size = load_dict(NC_vs_size_filename)
		NC_vs_evolv = load_dict(NC_evolv_filename)
		NC_list = [NC for NC in NC_vs_size if NC > 0]
		ax[i//5, i%5].scatter([np.log10(NC_vs_size[NC]) for NC in NC_list], [np.log10(NC_vs_evolv[NC]) if NC_vs_evolv[NC] > 0 else np.nan for NC in NC_list], s=3, c='r', linestyle='-', zorder=0)
		ax[i//5, i%5].plot([-0.1, 8], [np.log10(map_vs_number_pheno[type_structure] - 1),]*2, c='k', linestyle=':', zorder=-2)
		ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])
		print(type_structure, 'max evolv compared to max', map_vs_number_pheno[type_structure] - 1 - max([e for e in NC_vs_evolv.values()]))
		if 'Fibonacci' in type_structure:
			length_coding_region = np.arange(0, L)
			size = [(K ** (L - l - 1)) for l in length_coding_region]
			if 'null' not in type_structure:
				evolv = [(K - 1) * l + (K - 1) * sum([(K-1)**lnc for lnc in range(0, L - l - 1 )]) for l in length_coding_region] #new coding region can be up to L -l -2
			else:
				evolv = [(K - 2) * l for l in length_coding_region]
			ax[i//5, i%5].plot(np.log10(size), np.log10(evolv), c='grey', zorder=-5)
		assert max([np.log10(NC_vs_size[NC]) for NC in NC_list]) < 7 #check axis limits
	ax[i//5, i%5].set_xlim(-0.03, 7)
	if i // 5 == 1:
		ax[i//5, i%5].set_xlabel(r'$\log_{10} |NC|$')
	if i % 5 == 0:
		ax[i//5, i%5].set_ylabel(r'$\log_{10} \epsilon_{NC}$') #+'\n'+r'$\epsilon_p = 0$ plotted as -1')
	[ax[i//5, i%5].annotate('ABCDEFGHIJ'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(9)]
	ax[9//5, 9%5].axis('off')

f.tight_layout()
f.savefig('./plots/evolv_vs_size.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('size distr in each GP map')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 5.5), sharex=True, sharey=True)
for i, type_structure in enumerate(param.all_types_structue):
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
	if isfile(NC_vs_size_filename):
		NC_vs_size = load_dict(NC_vs_size_filename)
		sorted_sizes = sorted([int(s) for NC, s in NC_vs_size.items() if NC > 0], reverse=True)
		unique_sizes = np.unique(sorted_sizes)
		ax[i//5, i%5].scatter([len([x for x in sorted_sizes if x >= s]) for s in unique_sizes], unique_sizes, s=3, c='grey', linestyle='-')
		ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])
		ax[i//5, i%5].set_xscale('log')
		ax[i//5, i%5].set_yscale('log')
	if i // 5 == 1:
		ax[i//5, i%5].set_xlabel(r'size rank $r$')
		
	if i % 5 == 0:
		 ax[i//5, i%5].set_ylabel(r'NC size $|NC|$')
	[ax[i//5, i%5].annotate('ABCDEFGHIJ'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(9)]
	ax[9//5, 9%5].axis('off')

f.tight_layout()
f.savefig('./plots/size_dist.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('assortativity in each GP map')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 5.5))
for i, type_structure in enumerate(param.all_types_structue):
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_network_filename = './data/NC_network'+type_structure+ '.gml'
	if isfile(NC_network_filename):
		NC_network_str_nodes = nx.read_gml(NC_network_filename)
		NC_network = nx.relabel_nodes(NC_network_str_nodes, mapping={n: int(n) for n in NC_network_str_nodes.nodes()}, copy=True)
		node_list = [n for n in NC_network.nodes()]
		degree_dict = dict(NC_network.degree(node_list))
		degree_list = [degree_dict[n] for n in node_list]
		degree_list_neighbours = [np.mean([degree_dict[n2] for n2 in NC_network.neighbors(n)]) for n in node_list]
		ax[i//5, i%5].scatter(degree_list, degree_list_neighbours, c='grey', alpha=0.4, s=2)
		ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure] + '\n' +'degree assortativ. {0:.2g}'.format(nx.degree_assortativity_coefficient(NC_network)))
		ax[i//5, i%5].set_xlabel('NC degree')
		ax[i//5, i%5].set_yscale('log')
		ax[i//5, i%5].set_xscale('log')
		ax[i//5, i%5].set_ylabel('mean neighbour\nNC degree')
		#lims = (0.8 * 10**int(np.log10(np.nanmin(degree_list_neighbours))), 1.2 * 10**np.ceil(np.log10(np.nanmax(degree_list_neighbours))))
		#ax[i//5, i%5].set_ylim(lims[0], lims[1])
	[ax[i//5, i%5].annotate('ABCDEFGHIJ'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(9)]
	ax[9//5, 9%5].axis('off')

f.tight_layout()
f.savefig('./plots/assortativity.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('degree dist in each GP map')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 5.5))
for i, type_structure in enumerate(param.all_types_structue):
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_network_filename = './data/NC_network'+type_structure+ '.gml'
	if isfile(NC_network_filename):
		NC_network_str_nodes = nx.read_gml(NC_network_filename)
		NC_network = nx.relabel_nodes(NC_network_str_nodes, mapping={n: int(n) for n in NC_network_str_nodes.nodes()}, copy=True)
		node_list = [n for n in NC_network.nodes()]
		NC_vs_degree = dict(NC_network.degree(node_list))

		sorted_degrees = sorted([int(x) for x in NC_vs_degree.values()], reverse=True)
		unique_degrees = np.unique(sorted_degrees)
		cum_dist = np.divide([len([x for x in sorted_degrees if x >= s]) for s in unique_degrees], len(NC_vs_degree))
		ax[i//5, i%5].scatter(unique_degrees, cum_dist, s=5, c='grey', linestyle='-')
		ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])# + '\n' +'{:.2e}'.format(len(NC_list)) + 'NCs')
		ax[i//5, i%5].set_ylabel('fraction of NCs\nwith degree k or greater')
		ax[i//5, i%5].set_xlabel('NC degree k')
		ax[i//5, i%5].set_yscale('log')
		ax[i//5, i%5].set_ylim(0.3* min(cum_dist), 3)
		ax[i//5, i%5].set_xscale('log')
	[ax[i//5, i%5].annotate('ABCDEFGHIJK'[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(9)]
	ax[9//5, 9%5].axis('off')

f.tight_layout()
f.savefig('./plots/degree_dist'+'.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('size vs evolv in each GP map -- include fit')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 5.5), sharex=True, sharey=True)
for i, type_structure in enumerate(param.all_types_structue):
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
	NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
	if isfile(NC_evolv_filename):
		NC_vs_size = load_dict(NC_vs_size_filename)
		NC_vs_evolv = load_dict(NC_evolv_filename)
		NC_list = [NC for NC in NC_vs_size if NC > 0]
		ax[i//5, i%5].scatter([np.log10(NC_vs_size[NC]) for NC in NC_list], [np.log10(NC_vs_evolv[NC]) if NC_vs_evolv[NC] > 0 else np.nan for NC in NC_list], s=3, c='r', linestyle='-', zorder=0)
		ax[i//5, i%5].plot([-0.1, 8], [np.log10(map_vs_number_pheno[type_structure] - 1),]*2, c='k', linestyle=':', zorder=-2)
		ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])
		###### fit
		min_e, max_e = max(1, min([e for e in NC_vs_evolv.values()])), max([e for e in NC_vs_evolv.values()])
		min_s, max_s = max(1, min([e for e in NC_vs_size.values()])), max([e for e in NC_vs_size.values()])
		beta = np.log(min_e/max_e)/np.log(min_s/max_s)
		print('evolvability-size power law', type_structure, beta)
		ax[i//5, i%5].plot([np.log10(NC_vs_size[NC]) for NC in NC_list], [np.log10(max_e * (NC_vs_size[NC]/max_s)**beta) for NC in NC_list], c='g')
		#####
		if 'Fibonacci' in type_structure:
			length_coding_region = np.arange(0, L)
			size = [(K ** (L - l - 1)) for l in length_coding_region]
			if 'null' not in type_structure:
				evolv = [(K - 1) * l + (K - 1) * sum([(K-1)**lnc for lnc in range(0, L - l - 1 )]) for l in length_coding_region] #new coding region can be up to L -l -2
			else:
				evolv = [(K - 2) * l for l in length_coding_region]
			ax[i//5, i%5].plot(np.log10(size), np.log10(evolv), c='grey', zorder=-5)
		assert max([np.log10(NC_vs_size[NC]) for NC in NC_list]) < 7 #check axis limits
	ax[i//5, i%5].set_xlim(-0.03, 7)
	if i // 5 == 1:
		ax[i//5, i%5].set_xlabel(r'$\log_{10} |NC|$')
	if i % 5 == 0:
		ax[i//5, i%5].set_ylabel(r'$\log_{10} \epsilon_{NC}$') #+'\n'+r'$\epsilon_p = 0$ plotted as -1')
	[ax[i//5, i%5].annotate('ABCDEFGHIJ'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(9)]
	ax[9//5, 9%5].axis('off')

f.tight_layout()
f.savefig('./plots/evolv_vs_size2.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('evolv per pheno vs per NC')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 5.5))
for i, type_structure in enumerate(param.all_types_structue):
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
	NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
	NCindex_vs_ph_filename = './data/NC_array'+ type_structure+ 'NCindex_vs_ph.csv'
	if isfile(NC_evolv_filename):
		NC_vs_size = load_dict(NC_vs_size_filename)
		NC_vs_evolv = load_dict(NC_evolv_filename)
		NCindex_vs_ph = load_dict(NCindex_vs_ph_filename)
		ph_vs_NCs = {}
		for NC, ph in NCindex_vs_ph.items():
			try:
				ph_vs_NCs[ph].append(NC)
			except KeyError:
				ph_vs_NCs[ph] = [NC,]
		ph_list = list(ph_vs_NCs.keys())
		ph_vs_highest_NCev = {ph: NC_vs_evolv[max(NC_list, key=NC_vs_size.get)] for ph, NC_list in ph_vs_NCs.items()}
		ph_vs_av_NCev = {ph: np.sum([NC_vs_evolv[NC]*NC_vs_size[NC] for NC in NC_list])/np.sum([NC_vs_size[NC] for NC in NC_list]) for ph, NC_list in ph_vs_NCs.items()}
		ax[i//5, i%5].scatter([ph_vs_av_NCev[ph] for ph in ph_list], [ph_vs_highest_NCev[ph] for ph in ph_list], s=5, c='r', linestyle='-', zorder=0)
		max_x = max(ph_vs_highest_NCev.values()) + 3
		ax[i//5, i%5].plot([0, max_x], [0, max_x], c='k', linestyle=':', zorder=-2)
		ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])
		ax[i//5, i%5].set_xlim(0, max_x)
		ax[i//5, i%5].set_xlim(0, max_x)
		ax[i//5, i%5].set_xlabel(r'expected $\epsilon_{NC}$ for a NC chosen'+'\n'+'at random with probability $\propto$|NC|'+'\nfrom NCs mapping to phenotype p')
		ax[i//5, i%5].set_ylabel(r'$\epsilon_{NC}$ of the largest NC'+'\nfor phenotype p') #+'\n'+r'$\epsilon_p = 0$ plotted as -1')
	[ax[i//5, i%5].annotate('ABCDEFGHIJ'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(9)]
	ax[9//5, 9%5].axis('off')

f.tight_layout()
f.savefig('./plots/evolv_neutral_set.png', bbox_inches='tight', dpi=200)
