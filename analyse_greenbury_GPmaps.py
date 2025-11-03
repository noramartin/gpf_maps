#!/usr/bin/env python3
import numpy as np 
import networkx as nx 
import itertools
from os.path import isfile
from functools import partial
import pandas as pd
import sys
import random
import parameters as param
from functions.navigability_functions import *
from functions.fibonacci_functions import get_Fibonacci_GPmap, neighbours_g_not_stopcodon
####################################################################################################################################################
####################################################################################################################################################
type_structure = sys.argv[1]
print(type_structure)
K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
if 'null' not in type_structure:
	neighbors_function = partial(neighbours_g, K=K, L=L)
else:
	neighbors_function = partial(neighbours_g_not_stopcodon, K=K, L=L)
####################################################################################################################################################
for exclude_zero_evolv in [False, True]:
	if exclude_zero_evolv:
		filename_addition = 'exclude_zero_evolv'
	else:
		filename_addition = ''
	####################################################################################################################################################
	print('get GP map')
	####################################################################################################################################################
	GPmapfilename = './data/GPmap'+ type_structure+ '.npy'
	if not isfile(GPmapfilename):
		if 'Fibonacci' not in type_structure:
			GPmap = read_in_Greenbury_data('gp_maps/'+type_structure+'/'+param.type_structure_vs_filename[type_structure]+'.txt', K, L)
		else:
			GPmap = get_Fibonacci_GPmap(K, L)
		np.save(GPmapfilename, GPmap)
	else:
		GPmap = np.load(GPmapfilename)
	phs, counts = np.unique(GPmap, return_counts=True)
	ph_vs_size = {ph: count for ph, count in zip(phs, counts) if ph > 0}
	####################################################################################################################################################
	print('get NC array')
	####################################################################################################################################################
	NC_array, NCindex_vs_ph = find_all_NCs(GPmap, neighbors_function, filename='./data/NC_array'+ type_structure, highly_fragmented=False)
	NCs, counts = np.unique(NC_array, return_counts=True)
	NC_vs_size = {NCindex: count for NCindex, count in zip(NCs, counts) if NCindex > 0}
	if type_structure == 'HP_20':
		assert len(NC_vs_size) == 6586
	save_dict(NC_vs_size, './data/NC_array'+ type_structure +'NC_vs_size.csv')
	ph_vs_NClist = get_ph_vs_NCs(NCindex_vs_ph)
	####################################################################################################################################################
	print('get NC graph')
	####################################################################################################################################################
	NC_network_filename = './data/NC_network'+type_structure+filename_addition+ '.gml'
	if isfile(NC_network_filename):
		NC_network_str_nodes = nx.read_gml(NC_network_filename)
		NC_network = nx.relabel_nodes(NC_network_str_nodes, mapping={n: int(n) for n in NC_network_str_nodes.nodes()}, copy=True)
	else:
		NC_network = get_NCgraph(NC_array, neighbors_function, exclude_zero_evolv=exclude_zero_evolv)
		NC_network_str_nodes = nx.relabel_nodes(NC_network, mapping={n: str(n) for n in NC_network.nodes()}, copy=True)
		nx.write_gml(NC_network_str_nodes, NC_network_filename)
	NC_list = [NC for NC in NC_network.nodes()]
	assert len(NC_list) == len(NC_vs_size.keys()) or exclude_zero_evolv
	phenotypes_list = [p for p in np.unique(GPmap) if not isundefinedpheno(p)]
	del NC_network_str_nodes
	####################################################################################################################################################
	print('get ph evovability')
	####################################################################################################################################################
	if not exclude_zero_evolv:
		ph_vs_neighbours = {ph: set([NCindex_vs_ph[NC2] for NC in ph_vs_NClist[ph] for NC2 in NC_network.neighbors(NC)]) for ph in phenotypes_list}
		ph_vs_evolv = {ph: len(ph_vs_neighbours[ph]) for ph in phenotypes_list}
		save_dict(ph_vs_evolv, './data/ph_vs_evolv'+type_structure+filename_addition+ '.csv')
	####################################################################################################################################################
	print('NC evolvabilities')
	####################################################################################################################################################
	NC_evolv_filename = './data/NC_evolv_'+type_structure+filename_addition+ '.csv'
	if not isfile(NC_evolv_filename):
		NC_vs_evolv = {NC: len(set([NCindex_vs_ph[NC2] for NC2 in NC_network.neighbors(NC)])) for NC in NC_list}
		if exclude_zero_evolv:
			assert min(list(NC_vs_evolv.values())) > 0
		save_dict(NC_vs_evolv, NC_evolv_filename)
	else:
		NC_vs_evolv = load_dict(NC_evolv_filename)
	for distribution in ['uniform', 'exponential']:
		####################################################################################################################################################
		print('peaks analysis - need number of peaks and heights', distribution)
		####################################################################################################################################################
		total_peak_number_filename = './data/total_peak_number_'+type_structure+filename_addition+'_iterations'+str(param.iterations_peaks)+ distribution+'.npy'
		total_peak_size_filename = './data/total_peak_size_'+type_structure+filename_addition+'_iterations'+str(param.iterations_peaks)+ distribution+'.npy'
		peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+filename_addition+'_iterations'+str(param.iterations_peaks)+ distribution+'.csv'
		peak_heights_reached_evolv = './data/peak_heights_reached_evolv_'+type_structure+filename_addition+'_pfmaps'+str(param.no_pf_maps_evolutionary_walks)+'_nruns'+str(param.no_runs_evolutionary_walks)+'pop_size'+str(param.pop_size_evolutionary_walks)+ distribution+'.csv'
		if not isfile(peak_evolv_height_filename) or not isfile(total_peak_size_filename) or not isfile(total_peak_number_filename) or not (isfile(peak_heights_reached_evolv) and not exclude_zero_evolv):
			total_peak_number_list, total_peak_size_list, peak_evolv_list, peak_height_list, iteration_list, peak_size_list, peak_NC_list, pf_map_list = count_peaks(NC_network, NCindex_vs_ph, NC_vs_evolv, NC_vs_size, 
				                                                                                                                                                     phenotypes_list, distribution, param.iterations_peaks, return_pf_maps =True, save_heights_up_to_iteration=param.no_pf_maps_evolutionary_walks)
			np.save(total_peak_size_filename, total_peak_size_list)
			np.save(total_peak_number_filename, total_peak_number_list)
			df_peak_evolv_height = pd.DataFrame.from_dict({'iteration': iteration_list, 'peak evolvability': peak_evolv_list, 'peak height': peak_height_list, 'peak size': peak_size_list, 'peak NC': peak_NC_list})
			df_peak_evolv_height.to_csv(peak_evolv_height_filename)	
			if not exclude_zero_evolv:
				peak_heights_reached_evol_df = peak_heights_reached_evolv_random_walk(NC_array, pf_map_list[:param.no_pf_maps_evolutionary_walks], N=param.pop_size_evolutionary_walks, iterations=param.no_runs_evolutionary_walks,
																					  NCindex_vs_ph=NCindex_vs_ph, NC_network=NC_network, neighbors_function=neighbors_function, NC_vs_size=NC_vs_size)
				peak_heights_reached_evol_df.to_csv(peak_heights_reached_evolv)	

				
	####################################################################################################################################################
	print('navigability analysis')
	####################################################################################################################################################
	navigability_filename = './data/navigability'+type_structure+filename_addition+'_iterations'+str(param.iterations_nav)+'number_source_target_pairs'+str(param.number_source_target_pairs)+'.npy'
	if not isfile(navigability_filename):
		if exclude_zero_evolv:
			ph_vs_NClist2 = {ph: [NC for NC in l if NC in NC_network] for ph, l in ph_vs_NClist.items()}
		else:
			ph_vs_NClist2 = ph_vs_NClist
		navigability = find_navigability(param.iterations_nav, param.number_source_target_pairs, NC_network, ph_vs_NClist2, NCindex_vs_ph, NC_vs_size)
		np.save(navigability_filename, np.array([navigability,]))
	####################################################################################################################################################
	print('navigability analysis for the largest network component')
	####################################################################################################################################################
	navigability_filename = './data/largest_component_navigability'+type_structure+filename_addition+'_iterations'+str(param.iterations_nav)+'number_source_target_pairs'+str(param.number_source_target_pairs)+'.npy'
	if not exclude_zero_evolv:
		largest_component_network = get_largest_component(NC_network)
		NC_vs_evolv_component = {NC: len(set([NCindex_vs_ph[NC2] for NC2 in largest_component_network.neighbors(NC)])) for NC in largest_component_network.nodes()}
		assert min([e for e in NC_vs_evolv_component.values()]) >= 1
		ph_vs_NClist_component = {ph: [n for n in nlist if n in largest_component_network.nodes()] for ph, nlist in ph_vs_NClist.items()}
		NCindex_vs_ph_component= {n:ph for n, ph in NCindex_vs_ph.items() if n in largest_component_network.nodes()}
		NC_vs_size_component= {n:s for n, s in NC_vs_size.items() if n in largest_component_network.nodes()}
		navigability = find_navigability(param.iterations_nav, param.number_source_target_pairs, largest_component_network, ph_vs_NClist_component, NCindex_vs_ph_component, NC_vs_size_component, all_connected=True)
		np.save(navigability_filename, np.array([navigability,]))
		###
		NC_list_to_include_component = [max(NC_list, key=NC_vs_size.get) for ph, NC_list in ph_vs_NClist_component.items() if len(NC_list) > 0]
		NC_vs_evolv2_component = {NC: len(set([NCindex_vs_ph[NC2] for NC2 in largest_component_network.neighbors(NC) if NC2 in NC_list_to_include_component])) for NC in NC_list_to_include_component}
		save_dict(NC_vs_evolv2_component, './data/largest_component_NC_evolv_'+type_structure+filename_addition+ '_oneNC_per_pheno.csv')
		###
		ph_vs_evolv_component = {ph: len(set([NCindex_vs_ph[NC2] for NC in ph_vs_NClist_component[ph] for NC2 in largest_component_network.neighbors(NC)])) for ph in phenotypes_list if len(ph_vs_NClist_component[ph]) > 0}
		save_dict(ph_vs_evolv_component, './data/largest_component_ph_vs_evolv'+type_structure+filename_addition+ '.csv')
	####################################################################################################################################################
	print('data for comparison in reduced-dimensionality plot')
	####################################################################################################################################################
	NC_evolv_filename = './data/NC_evolv_'+type_structure+filename_addition+ '_summary.csv'
	if not isfile(NC_evolv_filename):
		pd.DataFrame.from_dict({'NC': NC_list, 
								'evolv': [NC_vs_evolv[NC] for NC in NC_list ], 
								'size': [NC_vs_size[NC] for NC in NC_list]}).to_csv(NC_evolv_filename)
	####################################################################################################################################################
	print('get evolvability if only one NC per pheno is retained (largest)')
	####################################################################################################################################################
	NC_evolv_filename2 = './data/NC_evolv_'+type_structure+filename_addition+ '_oneNC_per_pheno.csv'
	if not isfile(NC_evolv_filename2) and not exclude_zero_evolv:
		NC_list_to_include = [max(NC_list, key=NC_vs_size.get) for ph, NC_list in ph_vs_NClist.items()]
		NC_vs_evolv2 = {NC: len(set([NCindex_vs_ph[NC2] for NC2 in NC_network.neighbors(NC) if NC2 in NC_list_to_include])) for NC in NC_list_to_include}
		for NC in NC_list_to_include:
			assert len(set([NCindex_vs_ph[NC2] for NC2 in NC_network.neighbors(NC) if NC2 in NC_list_to_include])) == len([NCindex_vs_ph[NC2] for NC2 in NC_network.neighbors(NC) if NC2 in NC_list_to_include]) # only have one NC per pheno
		save_dict(NC_vs_evolv2, NC_evolv_filename2)

if 'null' in type_structure:
	assert 1 == 2 # rest not implemented
number_NCs_unperturbed = len(NC_vs_size.keys())

####################################################################################################################################################
####################################################################################################################################################
print('repeat analysis for limited dimensionality in GP map')
####################################################################################################################################################
####################################################################################################################################################

for dimensionality in np.arange(1, L):#[::-1]:
	print('dimensionality', dimensionality)
	sites_to_mutate = np.random.choice(L, size=dimensionality, replace=False)
	assert len(set(sites_to_mutate)) == dimensionality and max(sites_to_mutate) < L
	type_structure_dim = type_structure + 'dim' + str(dimensionality)
	neighbors_function = partial(neighbours_g_sites, K=K, L=L, sites_to_mutate=sites_to_mutate)
	if not isfile('./data/total_peak_size_'+type_structure_dim+'_iterations'+str(param.iterations_peaks)+ distribution+'.npy'):
		###
		if dimensionality < L-2 and (type_structure != 's_2_8' and dimensionality > 2):
			NC_array, NCindex_vs_ph = find_all_NCs(GPmap, neighbors_function, filename='./data/NC_array'+ type_structure_dim, highly_fragmented=True)
		else:
			NC_array, NCindex_vs_ph = find_all_NCs(GPmap, neighbors_function, filename='./data/NC_array'+ type_structure_dim)
		NCs, counts = np.unique(NC_array, return_counts=True)
		NC_vs_size = {NCindex: count for NCindex, count in zip(NCs, counts) if NCindex > 0}
		if dimensionality < L and not len(NC_vs_size.keys()) > number_NCs_unperturbed:
			print('no NCs fragmented', type_structure, 'dimensionality', dimensionality)
		####
		NC_network_filename = './data/NC_network'+type_structure_dim+ '.gml'
		if isfile(NC_network_filename):
			NC_network_str_nodes = nx.read_gml(NC_network_filename)
			NC_network = nx.relabel_nodes(NC_network_str_nodes, mapping={n: int(n) for n in NC_network_str_nodes.nodes()}, copy=True)
			NC_list = [NC for NC in NC_network.nodes()]
		else:
			NC_network = get_NCgraph(NC_array, neighbors_function, exclude_zero_evolv=False)
			NC_network_str_nodes = nx.relabel_nodes(NC_network, mapping={n: str(n) for n in NC_network.nodes()}, copy=True)
			nx.write_gml(NC_network_str_nodes, NC_network_filename)
			NC_list = [NC for NC in NC_network.nodes()]
			assert len([NC for NC in NC_list for NC2 in NC_network.neighbors(NC) if NCindex_vs_ph[NC] == NCindex_vs_ph[NC2]]) == 0 #no adjacant NCs of the same phenotype if neighbour function used consistently
		phenotypes_list = [p for p in np.unique(GPmap) if not isundefinedpheno(p)]
		NC_vs_evolv = {NC: len(set([NCindex_vs_ph[NC2] for NC2 in NC_network.neighbors(NC)])) for NC in NC_list}
		####################################################################################################################################################
		print('NC evolvabilities')
		####################################################################################################################################################
		NC_evolv_filename = './data/NC_evolv_'+type_structure_dim+ '.csv'
		if not isfile(NC_evolv_filename):
			pd.DataFrame.from_dict({'NC': NC_list, 
									'evolv': [NC_vs_evolv[NC] for NC in NC_list ], 
									'size': [NC_vs_size[NC] for NC in NC_list]}).to_csv(NC_evolv_filename)
		for distribution in ['uniform']:
			####################################################################################################################################################
			print('peaks analysis - need number of peaks and heights')
			####################################################################################################################################################
			total_peak_size_filename = './data/total_peak_size_'+type_structure_dim+'_iterations'+str(param.iterations_peaks)+ distribution+'.npy'
			if not isfile(total_peak_size_filename):
				total_peak_number_list, total_peak_size_list, peak_evolv_list, peak_height_list, iteration_list, peak_size_list, peak_NC_list = count_peaks(NC_network, NCindex_vs_ph, NC_vs_evolv, NC_vs_size, phenotypes_list, distribution, param.iterations_peaks, save_heights_up_to_iteration=1)
				np.save(total_peak_size_filename, total_peak_size_list)

####################################################################################################################################################
####################################################################################################################################################
print('repeat analysis for permuted GP map')
####################################################################################################################################################
####################################################################################################################################################
neighbors_function = partial(neighbours_g, K=K, L=L)
type_structure_dim = type_structure + 'permuted'
####################################################
GPmap_permuted_filename = './data/GPmap_permuted'+type_structure_dim+ '.npy'
if isfile(GPmap_permuted_filename):
	GPmap_permuted = np.load(GPmap_permuted_filename)
else:
	GPmap_permuted = create_uncorrelated_GPmap(GPmap)
	np.save(GPmap_permuted_filename, GPmap_permuted)			
###
if not type_structure == 's_2_8':
	NC_array, NCindex_vs_ph = find_all_NCs(GPmap_permuted, neighbors_function, filename='./data/NC_array'+ type_structure_dim, highly_fragmented=True)
else:
	NC_array, NCindex_vs_ph = find_all_NCs(GPmap_permuted, neighbors_function, filename='./data/NC_array'+ type_structure_dim, highly_fragmented=False)	
NCs, counts = np.unique(NC_array, return_counts=True)
NC_vs_size = {NCindex: count for NCindex, count in zip(NCs, counts) if NCindex > 0}
assert len(NC_vs_size.keys()) > number_NCs_unperturbed
####
NC_network_filename = './data/NC_network'+type_structure_dim+ '.gml'
if isfile(NC_network_filename):
	NC_network_str_nodes = nx.read_gml(NC_network_filename)
	NC_network = nx.relabel_nodes(NC_network_str_nodes, mapping={n: int(n) for n in NC_network_str_nodes.nodes()}, copy=True)
else:
	NC_network = get_NCgraph(NC_array, neighbors_function, exclude_zero_evolv=False)
	NC_network_str_nodes = nx.relabel_nodes(NC_network, mapping={n: str(n) for n in NC_network.nodes()}, copy=True)
	nx.write_gml(NC_network_str_nodes, NC_network_filename)
NC_list = [NC for NC in NC_network.nodes()]
phenotypes_list = [p for p in np.unique(GPmap) if not isundefinedpheno(p)]
NC_vs_evolv = {NC: len(set([NCindex_vs_ph[NC2] for NC2 in NC_network.neighbors(NC)]))for NC in NC_list}
####################################################################################################################################################
print('NC evolvabilities')
####################################################################################################################################################
NC_evolv_filename = './data/NC_evolv_'+type_structure_dim+ '.csv'
if not isfile(NC_evolv_filename):
	pd.DataFrame.from_dict({'NC': NC_list, 
								'evolv': [NC_vs_evolv[NC] for NC in NC_list ], 
								'size': [NC_vs_size[NC] for NC in NC_list]}).to_csv(NC_evolv_filename)
for distribution in ['uniform']:
	####################################################################################################################################################
	print('peaks analysis - need number of peaks and heights')
	####################################################################################################################################################
	total_peak_size_filename = './data/total_peak_size_'+type_structure_dim+'_iterations'+str(param.iterations_peaks)+ distribution+'.npy'
	if not isfile(total_peak_size_filename):
		total_peak_number_list, total_peak_size_list, peak_evolv_list, peak_height_list, iteration_list, peak_size_list, peak_NC_list = count_peaks(NC_network, NCindex_vs_ph, NC_vs_evolv, NC_vs_size, phenotypes_list, distribution, param.iterations_peaks, save_heights_up_to_iteration=1)
		np.save(total_peak_size_filename, total_peak_size_list)
