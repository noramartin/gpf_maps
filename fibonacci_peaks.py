#!/usr/bin/env python3
import numpy as np 
import networkx as nx 
from copy import deepcopy
import itertools
from os.path import isfile
from functools import partial
from scipy.stats import gmean, skew
import pandas as pd
import sys
from collections import Counter
from functions.navigability_functions import find_navigability, neighbours_g
from functions.fibonacci_functions import *
import parameters as param

print(sys.argv)
L, K = int(sys.argv[1]), int(sys.argv[2]) #6, 3
type_analysis = int(sys.argv[3]) # -1 == test, 3== paths counting, 4==navigability for reduced maps, 5=synthetic networks
####################################################################################################################################################
print('get network of phenotypes')
####################################################################################################################################################
GPmapfilename = './data/fibonacciGPmap_L'+str(L) +'_'+str(K)+ '.npy'
if not isfile(GPmapfilename):
    GPmap = get_Fibonacci_GPmap(K, L)
    np.save(GPmapfilename, GPmap)
else:
	GPmap = np.load(GPmapfilename)
phs, counts = np.unique(GPmap, return_counts=True)
ph_vs_size = {ph: count for ph, count in zip(phs, counts) if ph > 0}
####################################################################################################################################################
print('quick test of code')
####################################################################################################################################################
if type_analysis == -1:
	for type_model in ['standard', 'low_evolvability']:
		print(type_model)
		if type_model.startswith('standard'):
			neighbours_function = partial(neighbours_g, K=K, L=L)
		elif type_model.startswith('low_evolvability'):
			neighbours_function = partial(neighbours_g_not_stopcodon, K=K, L=L)
		####################################################################################################################################################
		print('get network of phenotypes (in this case equals NCs)')
		####################################################################################################################################################
		phenotype_network_filename = './data/fibonacciGPmap_L'+str(L)+'_'+str(K) +type_model+ '.gml'
		if isfile(phenotype_network_filename):
			phenotype_network_str_nodes = nx.read_gml(phenotype_network_filename)
			phenotype_network = nx.relabel_nodes(phenotype_network_str_nodes, mapping={n: int(n) for n in phenotype_network_str_nodes.nodes()}, copy=True)
			phenotpes_list = [ph for ph in phenotype_network.nodes()]
			assert min(phenotpes_list) > 0
		else:
			phenotpes_list = [ph for ph in np.unique(GPmap) if ph > 0]
			phenotype_network = make_phenotype_network(GPmap, neighbours_function, phenotpes_list=phenotpes_list)
			assert 0 not in phenotype_network.nodes()
			phenotype_network_str_nodes = nx.relabel_nodes(phenotype_network, mapping={n: str(n) for n in phenotype_network.nodes()}, copy=True)
			nx.write_gml(phenotype_network_str_nodes, phenotype_network_filename)
		for p in phenotpes_list:
			p_str = pheno_int_to_str(p, K)
			number_neighbours = len([n for n in phenotype_network.neighbors(p)])
			if type_model == 'low_evolvability':
				expected_number_neighbours = len(p_str) * (K - 2)
			elif type_model == 'standard':
				len_nc = L - len(p_str) - 1 # position of current stop codon substracted
				number_phenos_after_sc_mut = (K - 1) * sum([(K-1)**l for l in range(0, len_nc)])
				expected_number_neighbours = len(p_str) * (K - 1) +number_phenos_after_sc_mut
			assert expected_number_neighbours == number_neighbours
			for p2 in phenotype_network.neighbors(p):
				p2_str = pheno_int_to_str(p2, K)
				if type_model == 'standard': #can either be the same length and differ in a single site, or different length and be exact up to joint end
					if len(p_str) == len(p2_str):
						assert Hamming_dist(p_str, p2_str) == 1
					else:
						min_len = min(len(p_str), len(p2_str))
						assert Hamming_dist(p_str[:min_len], p2_str[:min_len]) == 0
				elif type_model == 'low_evolvability':
					assert len(p_str) == len(p2_str)
					assert Hamming_dist(p_str, p2_str) == 1
		print(type_model, 'tests passed for network with ', len(phenotype_network.edges()))
		####################################################################################################################################################
		print('for a few PF maps, check peak calculation in geno space')
		####################################################################################################################################################
		for rep in range(param.iterations_nav):
			pf_map = {ph: f for ph, f in zip(phenotpes_list, np.random.uniform(size=len(phenotpes_list)))}
			pf_map[0] = 0
			geno_graph = make_genotype_network_tests(GPmap, neighbours_function, pf_map)
			peaks_identity_list = find_peak_identities(phenotype_network, pheno_vs_fitness=pf_map)
			#### test if peaks in the NC map are also peaks in the genotype map
			shortest_paths_geno_graph = dict(nx.all_pairs_shortest_path_length(geno_graph))
			phenotype_phenotype_connections_from_geno_graph = {p: set([]) for p in phenotpes_list + [0,]}
			for g in shortest_paths_geno_graph:
				for g2 in shortest_paths_geno_graph[g]:
					p, p2 = GPmap[tuple([int(x) for x in g])], GPmap[tuple([int(x) for x in g2])]
					if p != p2:
						phenotype_phenotype_connections_from_geno_graph[p].add(p2)
			peaks_geno_graph = [p for p, l in phenotype_phenotype_connections_from_geno_graph.items() if len(l) == 0 and p != 0]
			assert len(peaks_geno_graph) == len(peaks_identity_list)
			for p in peaks_identity_list:
				assert p in peaks_geno_graph
		####################################################################################################################################################
		print('for a few PF maps, check navigability calculation in geno space')
		####################################################################################################################################################
		mean_nav = 0
		for rep in range(param.iterations_nav):	
			pf_map = {ph: f for ph, f in zip(phenotpes_list, np.random.uniform(0, 1, size=len(phenotpes_list)))}
			pf_map[0] = 0
			phenotype_network_directed_edges = get_directed_network_edges(phenotype_network, pheno_vs_fitness = pf_map)
			print('repetition nav', rep)
			for rep2 in range(param.number_source_target_pairs):
				source = np.random.choice(phenotpes_list)
				target = np.random.choice([p for p in phenotpes_list if p != source])
				pf_map2 = deepcopy(pf_map)
				pf_map2[target] = 1
				#### test if path in NC graph
				phenotype_network_directed_st = nx.DiGraph()
				phenotype_network_directed_st.add_nodes_from([source, target])
				phenotype_network_directed_st.add_edges_from([e for e in phenotype_network_directed_edges if target not in e])
				for n in phenotype_network.neighbors(target):
					assert pf_map2[target] > pf_map2[n]
					phenotype_network_directed_st.add_edge(n, target)
				path_NC_graph = nx.has_path(phenotype_network_directed_st, source, target)
				### test if path in geno graph
				geno_graph = make_genotype_network_tests(GPmap, neighbours_function, pf_map2)
				g_source = np.random.choice([''.join([str(x) for x in g]) for g, ph in np.ndenumerate(GPmap) if ph == source])
				g_target = np.random.choice([''.join([str(x) for x in g]) for g, ph in np.ndenumerate(GPmap) if ph == target])
				path_geno_graph = nx.has_path(geno_graph, g_source, g_target)
				assert path_NC_graph == path_geno_graph
				if path_geno_graph:
					mean_nav += 1/(param.iterations_nav*param.number_source_target_pairs)
		print('finished nav tests, psi=', mean_nav, 'NC graph edges', len([e for e in phenotype_network_directed_edges]))
		####################################################################################################################################################
		print('for a few PF maps, check navigability calculation in geno space where phenos deleted')
		####################################################################################################################################################
		np.random.seed(2)
		nph = len(phenotpes_list)
		for permill_to_keep in [1, 50, 100, 500, 'all']:
			mean_nav = 0
			phenotype_network = make_phenotype_network_deleted_nodes(permill_to_keep, GPmap, neighbours_function, phenotype_network_filename='')
			phenotpes_list = [int(n) for n in phenotype_network.nodes()]
			assert 0 not in phenotpes_list and (permill_to_keep == 'all' or len(phenotpes_list) <= int(round(permill_to_keep/1000* nph)))
			if len(phenotpes_list) < 3:
				continue
			for rep in range(param.iterations_nav):
				pf_map = {ph: f for ph, f in zip(phenotpes_list, np.random.uniform(0, 1, size=len(phenotpes_list)))}
				phenotype_network_directed_edges = get_directed_network_edges(phenotype_network, pheno_vs_fitness = pf_map)
				pf_map[0] = 0
				for ph in np.unique(GPmap):
					if ph not in phenotpes_list: #deleted pheno
						pf_map[ph] = 0
				for rep2 in range(param.number_source_target_pairs):
					source = np.random.choice(phenotpes_list)
					target = np.random.choice([p for p in phenotpes_list if p != source])
					#### test if path in NC graph
					phenotype_network_directed_st = nx.DiGraph()
					phenotype_network_directed_st.add_edges_from([e for e in phenotype_network_directed_edges if target not in e])
					for n in phenotype_network.neighbors(target):
						phenotype_network_directed_st.add_edge(n, target)					
					path_NC_graph = nx.has_path(phenotype_network_directed_st, source, target)
					### test if path in geno graph
					pf_map2 = deepcopy(pf_map)
					pf_map2[target] = 1
					assert len([ph for ph, f in pf_map2.items() if f > 0]) == len([ph for ph in phenotype_network_directed_st.nodes()]) #only non-isolated non-zero nodes in NC graph
					geno_graph = make_genotype_network_tests(GPmap, neighbours_function, pf_map2)
					g_source = np.random.choice([''.join([str(x) for x in g]) for g, ph in np.ndenumerate(GPmap) if ph == source])
					g_target = np.random.choice([''.join([str(x) for x in g]) for g, ph in np.ndenumerate(GPmap) if ph == target])
					path_geno_graph = nx.has_path(geno_graph, g_source, g_target)
					assert path_NC_graph == path_geno_graph
					if path_geno_graph:
						mean_nav += 1/(param.iterations_nav*param.number_source_target_pairs)
			print('finished nav tests deletions, psi=', mean_nav)
####################################################################################################################################################
print('number of paths vs length')
####################################################################################################################################################
if type_analysis == 3:
	max_path_length_to_compute = 100
	for type_model in ['low_evolvability', 'standard']:
		filename_paths = './data/fibonaccinetwork_phenotypes_L'+str(L) +'_'+str(K)+'_iterations'+str(param.iterations_nav)+'_'+str(param.number_source_target_pairs)+type_model+ 'paths'+str(max_path_length_to_compute)+'.csv'
		if not isfile(filename_paths):
			print('compute number of paths', filename_paths)
			df_pl_data = {'path length': [], 'number paths': []}
			####################################################################################################################################################
			print('get network of phenotypes (in this case equals NCs)')
			####################################################################################################################################################
			if type_model.startswith('standard'):
				neighbours_function = partial(neighbours_g, K=K, L=L)
			elif type_model.startswith('low_evolvability'):
				neighbours_function = partial(neighbours_g_not_stopcodon, K=K, L=L)
			phenotype_network_filename = './data/fibonacciGPmap_L'+str(L)+'_'+str(K) +type_model+ '.gml'
			if isfile(phenotype_network_filename):
				phenotype_network_str_nodes = nx.read_gml(phenotype_network_filename)
				phenotype_network = nx.relabel_nodes(phenotype_network_str_nodes, mapping={n: int(n) for n in phenotype_network_str_nodes.nodes()}, copy=True)
			else:
				phenotpes_list = [ph for ph in np.unique(GPmap) if ph > 0]
				phenotype_network = make_phenotype_network(GPmap, neighbours_function, phenotpes_list=phenotpes_list)
				phenotype_network_str_nodes = nx.relabel_nodes(phenotype_network, mapping={n: str(n) for n in phenotype_network.nodes()}, copy=True)
				nx.write_gml(phenotype_network_str_nodes, phenotype_network_filename)
			phenotype_network.remove_nodes_from([n for n in phenotype_network.nodes() if phenotype_network.degree[n] == 0])
			phenotpes_list = [ph for ph in phenotype_network.nodes()]
			####################################################################################################################################################
			print('get path lengths')
			####################################################################################################################################################
			for r in range(param.iterations_nav):
				pf_map = {ph: f for ph, f in zip(phenotpes_list, np.random.uniform(0, 1, size=len(phenotpes_list)))}
				phenotype_network_directed_edges = get_directed_network_edges(phenotype_network, pf_map)
				for r2 in range(param.number_source_target_pairs):
					source = np.random.choice(phenotpes_list)
					target = np.random.choice([p for p in phenotpes_list if p != source])
					phenotype_network_directed_st = nx.DiGraph()
					phenotype_network_directed_st.add_edges_from([e for e in phenotype_network_directed_edges if target not in e])
					phenotype_network_directed_st.add_node(source)
					phenotype_network_directed_st.add_node(target)
					for n in phenotype_network.neighbors(target):
						phenotype_network_directed_st.add_edge(n, target)
					pathlength_list = [len(p) for p in nx.all_simple_paths(phenotype_network_directed_st, source, target, cutoff=max_path_length_to_compute)] #all paths on a NC graph are simple paths since every non-neutral mutation is a step up in fiitness, so can never return to a node
					pathlength_counter = Counter(pathlength_list)
					for l in range(1, max_path_length_to_compute + 1):
						df_pl_data['path length'].append(l)
						df_pl_data['number paths'].append(pathlength_counter[l])
			df_pl = pd.DataFrame.from_dict(df_pl_data)
			df_pl.to_csv(filename_paths)



####################################################################################################################################################
print('deleting phenotypes')
####################################################################################################################################################
if type_analysis == 4:
	mutations_only = 'standard'
	percentage_to_keep_vs_mean_evolv, percentage_to_keep_vs_mean_nav, percentage_to_keep_vs_nph = {}, {}, {}
	for permill_to_keep in [1, 10, 100, 500, 'all']:
		type_model = mutations_only + '_reduced_phenos_permill_'+str(permill_to_keep)
		print(type_model)
		if type_model.startswith('standard'):
			neighbours_function = partial(neighbours_g, K=K, L=L)
		elif type_model.startswith('low_evolvability'):
			neighbours_function = partial(neighbours_g_not_stopcodon, K=K, L=L)
		####################################################################################################################################################
		print('get network of phenotypes (in this case equals NCs)')
		####################################################################################################################################################
		phenotype_network_filename = './data/fibonacciGPmap_L'+str(L) +'_'+str(K)+type_model+ '.gml'
		if isfile(phenotype_network_filename):
			phenotype_network_str_nodes = nx.read_gml(phenotype_network_filename)
			phenotype_network = nx.relabel_nodes(phenotype_network_str_nodes, mapping={n: int(n) for n in phenotype_network_str_nodes.nodes()}, copy=True)	
		else:
			phenotype_network = make_phenotype_network_deleted_nodes(permill_to_keep, GPmap, neighbours_function, phenotype_network_filename)
		assert len([n for n in phenotype_network if len(set([ph2 for ph2 in phenotype_network.neighbors(n)])) == 0]) == 0
		phenotpes_list = [n for n in phenotype_network if len(set([ph2 for ph2 in phenotype_network.neighbors(n)])) > 0]
		if len(phenotpes_list) <= 5:
			continue
		evolv_values = [len(set([ph2 for ph2 in phenotype_network.neighbors(ph)])) for ph in phenotpes_list]
		assert min(evolv_values) > 0
		percentage_to_keep_vs_mean_evolv[type_model] = gmean([e for e in evolv_values])
		phenotpes_list = [ph for ph in phenotype_network.nodes()]
		percentage_to_keep_vs_nph[type_model] = len(phenotpes_list)
		assert min(phenotpes_list) > 0
		####################################################################################################################################################
		print('get navigability in random phenotype-fitness assignment')
		####################################################################################################################################################
		nav_filename = './data/fibonaccinetwork_phenotypes_L'+str(L) +'_'+str(K)+'_iterations'+str(param.iterations_nav)+'_'+str(param.number_source_target_pairs)+type_model+ 'list_navigability.npy'
		if not isfile(nav_filename):
			navigability = find_navigability(param.iterations_nav, param.number_source_target_pairs, phenotype_network, {ph: [ph,] for ph in phenotpes_list}, {ph: ph for ph in phenotpes_list}, {ph: ph_vs_size[ph] for ph in phenotpes_list})
			np.save(nav_filename, np.array([navigability]))
			del phenotype_network
		else:
			navigability = np.mean(np.load(nav_filename))
		percentage_to_keep_vs_mean_nav[type_model] = np.mean(navigability)
	type_model_list = [p  for p in percentage_to_keep_vs_mean_evolv.keys()]
	number_pheno_list = [percentage_to_keep_vs_nph[p] for p in type_model_list]
	evolv_list = [percentage_to_keep_vs_mean_evolv[p] for p in type_model_list]
	nav_list = [percentage_to_keep_vs_mean_nav[p] for p in type_model_list]
	df_nav = pd.DataFrame.from_dict({'number of phenotypes': number_pheno_list, 'navigability': nav_list, 'evolvability': evolv_list})
	df_nav.to_csv('./data/fibonacci'+'_navigability_data_deleted_phenotypes_L'+str(L) +'_'+str(K)+'_iterations'+str(param.iterations_nav)+'_'+str(param.number_source_target_pairs)+ 'missing_phenos.csv')


####################################################################################################################################################
print('navigability analysis with artifical network')
####################################################################################################################################################
if type_analysis == 5:
	number_pheno_list, evolvability_list, nav_list, type_dist_list, coefvar_list, skew_list = [], [], [], [], [], []
	for reps in np.arange(50, step=2): 
		frac = 0.005*(reps+1) # the mean degree is set to this fraction times the number of phenotypes
		for nph in [50, 100, 200, 300, 500, 10**3]: 
			for type_dist in ['geometric', 'Poisson', 'constant']:
				print(frac, nph, type_dist)
				evolv = frac * nph
				if evolv < 1:
					continue
				if type_dist == 'geometric':
					deg_sequence = np.random.geometric(1/evolv, nph) 
				elif type_dist == 'Poisson':
					deg_sequence = np.random.poisson(evolv, nph) 
				elif type_dist == 'constant':
					deg_sequence = [int(round(evolv)),] * nph									
				if sum(deg_sequence)%2!=0:
					deg_sequence[0]+=1
				####################################################################################################################################################
				print('get network of phenotypes (in this case equals NCs)')
				####################################################################################################################################################
				phenotype_network = nx.configuration_model(deg_sequence, create_using=nx.Graph, seed=None)
				phenotype_network = nx.Graph(phenotype_network) #remove multi-edges
				phenotype_network.remove_edges_from(nx.selfloop_edges(phenotype_network)) #remove self-loops
				phenotype_network.remove_nodes_from([n for n in phenotype_network if phenotype_network.degree(n) == 0]) #remove zero-degree nodes
				if len(phenotype_network.edges()) > 0:
					phenotpes_list = [ph for ph in phenotype_network.nodes()]
					number_pheno_list.append(len(phenotpes_list))
					evolv_values = [len(set([ph2 for ph2 in phenotype_network.neighbors(ph) if ph2 != ph])) for ph in phenotpes_list]
					for ph in phenotpes_list:
						assert len(set([ph2 for ph2 in phenotype_network.neighbors(ph) if ph2 != ph])) == len([ph2 for ph2 in phenotype_network.neighbors(ph)])
					evolvability_list.append(gmean([e for e in evolv_values]))
					assert min(evolv_values) > 0
					####################################################################################################################################################
					print('get navigability in random phenotype-fitness assignment')
					####################################################################################################################################################
					nav = find_navigability(param.iterations_nav, param.number_source_target_pairs, phenotype_network, {ph: [ph,] for ph in phenotpes_list}, {ph: ph for ph in phenotpes_list}, {ph: 1 for ph in phenotpes_list})
					nav_list.append(nav)
					type_dist_list.append(type_dist)
					coefvar_list.append(np.std(evolv_values)/np.mean(evolv_values))
					skew_list.append(skew(evolv_values))
					###
					df_degree = pd.DataFrame.from_dict({'seq': evolv_values})
					df_degree.to_csv('./data/deg_sequence_'+str(type_dist)+'_'+str(nph)+'reps'+str(reps)+'.csv')

	df_nav = pd.DataFrame.from_dict({'evolvability': evolvability_list, 'number of phenotypes': number_pheno_list, 'type distribution': type_dist_list, 'navigability': nav_list, 'coefficient of variation': coefvar_list, 'skew': skew_list})
	df_nav.to_csv('./data/sythetic_network_navigability_data_iterations'+str(param.iterations_nav)+'_'+str(param.number_source_target_pairs)+ '.csv')
