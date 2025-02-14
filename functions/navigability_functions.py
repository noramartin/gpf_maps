#!/usr/bin/env python3
import numpy as np 
import networkx as nx 
from copy import deepcopy
import itertools
from os.path import isfile
import random
import pandas as pd

def save_dict(dict_to_save, filename):
	keys_list = [k for k in dict_to_save.keys()]
	df = pd.DataFrame.from_dict({'keys': keys_list, 'values': [dict_to_save[k] for k in keys_list]})
	df.to_csv(filename)

def load_dict(filename):
	df = pd.read_csv(filename)
	return {k: v for k, v in zip(df['keys'].tolist(), df['values'].tolist())}

####################################################################################################################################################

def get_largest_component(NC_graph):
		S = [NC_graph.subgraph(c).copy() for c in nx.connected_components(NC_graph)] # get component subgraph, from networkx documentation
		index_largest_component = max(np.arange(len(S)), key={i: len(c.nodes()) for i, c in enumerate(S)}.get)
		print('fraction of NCs in largest component', len(S[index_largest_component])/len(NC_graph))
		return S[index_largest_component]


def isundefinedpheno(ph):
	if ph == 0:
		return True 
	else:
		return False


def get_NCgraph(NC_array, neighbors_function, exclude_zero_evolv=False):
	NC_network = nx.Graph()
	for g, NC in np.ndenumerate(NC_array):
		if not isundefinedpheno(NC) and NC not in NC_network.nodes():
			NC_network.add_node(NC)
		if not isundefinedpheno(NC):
			for g2 in neighbors_function(g):
				NC2 = NC_array[g2]
				if NC2 != NC and not isundefinedpheno(NC2) and NC2 not in NC_network.neighbors(NC):
					NC_network.add_edge(NC, NC2)
	assert 0 not in NC_network.nodes()
	if exclude_zero_evolv:
		n_nodes = len(NC_network.nodes())
		NC_list = [NC for NC in NC_network if len([x for x in NC_network.neighbors(NC)]) == 0]
		NC_network.remove_nodes_from(NC_list)
		assert n_nodes - len(NC_list) == len(NC_network.nodes())
	return NC_network


def find_all_NCs(GPmap, neighbors_function, filename=None, highly_fragmented=False):
	"""find individual neutral components for all non-unfolded structures in GP map;
	saves/retrieves information if filename given and files present; otherwise calculation from scratch;
	this part of code is from earlier projects"""
	K, L = GPmap.shape[1], GPmap.ndim
	if not isfile(filename+ '.npy') or not isfile(filename+ 'NCindex_vs_ph.csv'):
		NCindex_global_count = 1 #such that 0 remains for undefined structure
		seq_vs_NCindex_array = np.zeros((K,)*L, dtype='uint32')
		NCindex_vs_ph = {}
		for g, ph in np.ndenumerate(GPmap):
			if seq_vs_NCindex_array[g] == 0 and not isundefinedpheno(ph):
				if not highly_fragmented:
					genotype_set = connected_component(g, GPmap, neighbors_function)
				else:
					genotype_set = connected_component_fragmented(g, GPmap, neighbors_function)
				for g2 in genotype_set:
					seq_vs_NCindex_array[g2] = NCindex_global_count
				NCindex_vs_ph[NCindex_global_count] = ph
				NCindex_global_count += 1		
				assert NCindex_global_count < 2**32 - 1	
				print('NC finished', g, NCindex_global_count)	
		if filename:
			np.save(filename+ '.npy', seq_vs_NCindex_array)
			save_dict(NCindex_vs_ph, filename+ 'NCindex_vs_ph.csv')
	else:
		seq_vs_NCindex_array = np.load(filename + '.npy')
		NCindex_vs_ph = load_dict(filename+ 'NCindex_vs_ph.csv')
	return seq_vs_NCindex_array, NCindex_vs_ph

def connected_component(g0, GPmap, neighbors_function):
	#using the algorithm proposed in the appendix of Schaper's thesis; originally from W. Gruener et al., Monatsh Chem. 127, 375–389 (1996)
	U = np.zeros_like(GPmap, dtype='int8') #zero means not in set
	V = np.zeros_like(GPmap, dtype='int8') #zero means not in set
	V_list = []
	U_list = []
	pheno = int(GPmap[g0])
	U[g0] = 1
	U_list.append(deepcopy(g0))
	number_in_U = 1
	counter = 0
	while number_in_U > 0: #while there are elements in the unvisited list
		g1 = U_list.pop(0)
		assert GPmap[g1] == pheno and U[g1] == 1
		for g2 in neighbors_function(g1):
			if GPmap[g2] == pheno and U[g2] == 0 and V[g2] == 0:
				U[g2] = 1
				U_list.append(deepcopy(g2))
				number_in_U += 1
		U[g1] = 0
		number_in_U -= 1
		V[g1] = 1
		V_list.append(deepcopy(g1))
		counter += 1
		if counter % 10**4 == 0:
			print(str(counter) + ' genotypes evaluated')
			assert np.sum(U) == number_in_U == len(U_list)
			for g in U_list:
				assert U[g] == 1 and V[g] == 0
	print('finished component at g', g0)
	return V_list

def connected_component_fragmented(g0, GPmap, neighbors_function):
	#using the algorithm proposed in the appendix of Schaper's thesis; originally from W. Gruener et al., Monatsh Chem. 127, 375–389 (1996)
	V_list = []
	U_list = []
	pheno = int(GPmap[g0])
	U_list.append(deepcopy(g0))
	number_in_U = 1
	counter = 0
	while number_in_U > 0: #while there are elements in the unvisited list
		g1 = U_list.pop(0)
		assert GPmap[g1] == pheno
		for g2 in neighbors_function(g1):
			if GPmap[g2] == pheno and g2 not in U_list and g2 not in V_list:
				U_list.append(deepcopy(g2))
				number_in_U += 1
		number_in_U -= 1
		V_list.append(deepcopy(g1))
		counter += 1
		if counter % 10**4 == 0:
			print(str(counter) + ' genotypes evaluated')
			assert number_in_U == len(U_list)
	print('finished component at g', g0)
	return V_list

def neighbours_g_sites(g, K, L, sites_to_mutate):
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for pos in sites_to_mutate for new_K in range(K) if g[pos]!=new_K]


def get_ph_vs_NCs(NC_vs_ph):
	ph_vs_NClist = {}
	for NC, ph in NC_vs_ph.items():
		try:
			ph_vs_NClist[ph].append(NC)
		except KeyError:
			ph_vs_NClist[ph] = [NC,]
	return ph_vs_NClist

def neighbours_g(g, K, L): 
   """list all point mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for pos in range(L) for new_K in range(K) if g[pos]!=new_K]


def create_uncorrelated_GPmap(GPmap):
	K, L = GPmap.shape[1], GPmap.ndim
	phs, counts = np.unique(GPmap, return_counts=True)
	ph_vs_freq_incl_unfolded_unnormalised = {ph: count for ph, count in zip(phs, counts)}
	assert sum([c for c in ph_vs_freq_incl_unfolded_unnormalised.values()]) == K**L
	GPmap_permuted = np.zeros((K,)*L, dtype='uint32')
	list_phenos = [ph for ph, f in ph_vs_freq_incl_unfolded_unnormalised.items() for i in range(f)]
	assert len(list_phenos) == np.prod(GPmap.shape)
	random.shuffle(list_phenos)
	index = 0
	for g, ph in np.ndenumerate(GPmap):
		GPmap_permuted[g] = list_phenos[index]
		index += 1
	phs, counts = np.unique(GPmap_permuted, return_counts=True)
	for ph, c in zip(phs, counts):
		assert c == ph_vs_freq_incl_unfolded_unnormalised[ph]
	return GPmap_permuted


def read_in_Greenbury_data(filename_genos, K, L):
	list_phenotypes_duplicates_Greenbury = []
	with open(filename_genos, 'r') as fd:
		for l in fd:
			list_phenotypes_duplicates_Greenbury.append(int(l.strip()[:])) # he seems to have saved a structure index for each sequence, such that 1 is the first structure in the list and 0 is unfolded
	assert len(list_phenotypes_duplicates_Greenbury) == K**L
	###
	GPmap = np.zeros((K,)*L, dtype='uint32')
	for structureindex, ph in enumerate(list_phenotypes_duplicates_Greenbury):
		if structureindex == 0:
			g_str = '0' * L
		else:
			g_str = np.base_repr(structureindex, base = K, padding = L - len(np.base_repr(structureindex, base = K)))[::-1]
		assert len(g_str) == L
		g_tuple = tuple([int(x) for x in g_str])
		assert GPmap[g_tuple] == 0 #not yet allocated
		##test using Sam's example
		if structureindex == 12557963 and K == 4:
			assert g_tuple == (3, 2, 0, 2, 2, 3, 1, 2, 3, 3, 3, 2)
			if 'RNA_12' in filename_genos:
				assert ph == 56
		##
		if 's_2_8' in filename_genos and (ph == 0 or ph == 2):
			ph = {2: 0, 0: 2}[ph] #swap 0 and 2 because 2 is the undefined in the Polyomino data file 
		GPmap[g_tuple] = ph
	return GPmap
#######################################################################

def count_peaks(NC_network, NCindex_vs_ph, NC_vs_evolv, NC_vs_size, phenotpes_list, distribution, iterations_peaks):
	assert 0 not in phenotpes_list
	total_peak_size_list, peak_evolv_list, peak_height_list, iteration_list, peak_size_list = [], [], [], [], []
	for iteration in range(iterations_peaks):
		print('peak size iteration', iteration)
		total_size = 0
		if distribution == 'uniform':
			pheno_vs_fitness = {ph: f for ph, f in zip(phenotpes_list, np.random.uniform(size=len(phenotpes_list)))}
		elif distribution == 'exponential':
			pheno_vs_fitness = {ph: f for ph, f in zip(phenotpes_list, np.random.exponential(size=len(phenotpes_list)))}
		assert min([f for f in pheno_vs_fitness.values()]) > 0
		### number of peaks
		for NC in NC_network.nodes():
			fitness_NC = pheno_vs_fitness[NCindex_vs_ph[NC]]
			if max([pheno_vs_fitness[NCindex_vs_ph[NC2]] for NC2 in NC_network.neighbors(NC)]+[-1,]) < fitness_NC:
				if len(peak_height_list) < 5*10**6:
					peak_height_list.append(pheno_vs_fitness[NCindex_vs_ph[NC]])
					peak_evolv_list.append(NC_vs_evolv[NC])
					peak_size_list.append(NC_vs_size[NC])
					iteration_list.append(iteration)
				total_size += NC_vs_size[NC]
		total_peak_size_list.append(total_size)
	return total_peak_size_list, peak_evolv_list, peak_height_list, iteration_list, peak_size_list


#######################################################################

def find_navigability(iterations, number_source_target_pairs, NC_network, pheno_vs_NCs, NC_vs_ph, NC_vs_size, all_connected=False):
	if all_connected:
		print('including connectivity check')
	nav, nav_norm = 0, 0
	phenotpes_list = list([p for p in pheno_vs_NCs.keys() if len([NC for NC in pheno_vs_NCs[p] if NC in NC_network.nodes()]) > 0])
	for iteration in range(iterations):
		NC_network_directed_edges = get_directed_network_edges(NC_network, pheno_vs_NCs, NC_vs_ph)
		for source_target_count in range(number_source_target_pairs):
			source_ph = np.random.choice(phenotpes_list)
			target_ph = np.random.choice([p for p in phenotpes_list if p != source_ph])
			sourceNCs = [NC for NC in pheno_vs_NCs[source_ph] if NC in NC_network.nodes()]
			source = np.random.choice(sourceNCs, p = np.array([NC_vs_size[NC] for NC in sourceNCs])/sum([NC_vs_size[NC] for NC in sourceNCs]))
			NC_network_directed_st = make_network_given_target(NC_network_directed_edges, NC_network, target_ph, NC_vs_ph, pheno_vs_NCs)
			is_accessible = 0
			for target in pheno_vs_NCs[target_ph]:
				if all_connected: #working with simple connected component, so undirected graph needs to be connected
					assert nx.has_path(NC_network, source, target)
				if target in NC_network.nodes() and nx.has_path(NC_network_directed_st, source, target):
					is_accessible = 1
					break 		
			nav += is_accessible
			nav_norm += 1
	assert nav_norm == number_source_target_pairs * iterations
	print('navigability', nav/nav_norm)
	return nav/nav_norm

def get_directed_network_edges(NC_network, pheno_vs_NCs, NC_vs_ph):
	pheno_vs_fitness = {ph: f for ph, f in zip(list(pheno_vs_NCs.keys()), np.random.uniform(size=len(list(pheno_vs_NCs.keys()))))}
	assert min([f for f in pheno_vs_fitness.values()]) > 0 and max([f for f in pheno_vs_fitness.values()]) < 1
	NC_network_directed_edges = []
	for NC1, NC2 in NC_network.edges():
		if not pheno_vs_fitness[NC_vs_ph[NC1]] != pheno_vs_fitness[NC_vs_ph[NC2]]:
			print('adjacant edges same fitnes', NC1, NC2, pheno_vs_fitness[NC_vs_ph[NC1]], pheno_vs_fitness[NC_vs_ph[NC2]])
		assert pheno_vs_fitness[NC_vs_ph[NC1]] != pheno_vs_fitness[NC_vs_ph[NC2]]
		if pheno_vs_fitness[NC_vs_ph[NC1]] < pheno_vs_fitness[NC_vs_ph[NC2]]:
			NC_network_directed_edges.append((NC1, NC2))
		else:
			NC_network_directed_edges.append((NC2, NC1))
	return NC_network_directed_edges


def make_network_given_target(NC_network_directed_edges, NC_network, target_ph, NC_vs_ph, ph_vs_NClist):
	NC_network_directed_st = nx.DiGraph()
	NC_network_directed_st.add_nodes_from([NC for NC in NC_network.nodes()])
	NC_network_directed_st.add_edges_from([e for e in NC_network_directed_edges if target_ph not in [NC_vs_ph[e[0]], NC_vs_ph[e[1]]]])
	for target in ph_vs_NClist[target_ph]:
		if target in NC_network:
			NC_network_directed_st.add_node(target)
			for ph in NC_network.neighbors(target):
				NC_network_directed_st.add_edge(ph, target) #target is higher by definition (forced to one)	
			assert len([n for n in NC_network_directed_st.neighbors(target)]) == 0 # no upwards steps from target
	return NC_network_directed_st

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
	############################################################################################################
	#test mutations
	############################################################################################################
	K, L = 7, 12
	for dimensionality in range(1, L):
		sites_to_mutate = np.random.choice(L, size=dimensionality, replace=False)
		g = tuple(np.random.choice(K, size=L, replace=True))
		mutanted_g = neighbours_g_sites(g, K, L, sites_to_mutate)
		assert len(set(mutanted_g)) == len(mutanted_g) == (K - 1) * dimensionality
		mutanted_g2 = neighbours_g(g, K, L)
		assert len(set(mutanted_g2)) == len(mutanted_g2) == (K - 1) * L
		if dimensionality == L:
			for g2 in mutanted_g2:
				assert g2 in mutanted_g

