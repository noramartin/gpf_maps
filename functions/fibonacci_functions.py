import numpy as np 
import networkx as nx 
from copy import deepcopy

def get_Fibonacci_GPmap(K, L):
	GPmap = np.zeros((K,)*L, dtype='int32')
	for g, dummy in np.ndenumerate(GPmap):
		GPmap[g] = genotype_to_int_phenotype(g, K)
		assert GPmap[g] > 0 or K-1 not in g
	return GPmap


def genotype_to_int_phenotype(g, K):
	"""follows Weiss, Ahnert. 2018 J. R. Soc. Interface 15: 20170618"""
	assert max(g) <= K - 1
	if (K-1) not in g: #K-1 is the 'stop codon' - genotypes without stop codon are 'undefined'
		return 0
	ph = '1' + ''.join([str(x) for x in g]).split(str(K - 1))[0]
	ph_int = int(ph, base=max(2, K-1))
	assert ph_int < 2**31
	assert ph_int > 0
	return ph_int

def pheno_int_to_str(ph_int, K):
	str_ph = np.base_repr(ph_int, base = max(2, K-1))
	if len(str_ph) == 1:
		return ''
	return str_ph[1:]

def Hamming_dist(p1, p2):
	assert len(p1) == len(p2)
	return len([x for i, x in enumerate(p1) if x != p2[i]])

def neighbours_g_site(g, site, K): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=site else new_K for gpos, oldK in enumerate(g)]) for new_K in range(K) if g[site]!=new_K]

def neighbours_g_not_stopcodon(g, K, L):
	neighbours = []
	for site in range(L):
		if g[site] == (K-1) and (K-1) not in g[:site]:
			pass #do not mutate first stop codon
		elif (K-1) not in g[:site]:
			neighbours += deepcopy(neighbours_g_site(g, site, K - 1)) #cannot create a stop codon before first stop codon
		else:
			neighbours += deepcopy(neighbours_g_site(g, site, K)) #after first stop codon can do anything
	return neighbours

#########################################################################################################################

def make_phenotype_network(GPmap, neighbours_function, phenotpes_list=[]):
	if len(phenotpes_list) == 0:
		phenotpes_list = [ph for ph in np.unique(GPmap) if ph > 0]
	phenotype_network = nx.Graph()
	for g, ph in np.ndenumerate(GPmap):
		if ph in phenotpes_list and ph >= 0:
			if ph not in phenotype_network.nodes():
				phenotype_network.add_node(ph)
			neighbours = neighbours_function(g)
			for g2 in neighbours:
				ph2 = GPmap[g2]
				if ph2 != ph and ph2 in phenotpes_list:
					if ph >= 0 and ph2 >= 0 and ph2 not in phenotype_network.neighbors(ph):
						phenotype_network.add_edge(ph, ph2) 
	for ph in phenotype_network.nodes():
		assert ph in phenotpes_list
	for ph in phenotpes_list:
		assert ph in phenotype_network.nodes()
	return phenotype_network



def make_genotype_network_tests(GPmap, neighbours_function, pf_map):
	K, L = GPmap.shape[0], len(GPmap.shape)
	genotype_network = nx.MultiDiGraph()
	for g, ph in np.ndenumerate(GPmap):
		genotype_network.add_node(''.join([str(x) for x in g]))
		neighbours = neighbours_function(g)
		for g2 in neighbours:
			ph2 = GPmap[g2]
			if ph2 == ph:
				genotype_network.add_edge(''.join([str(x) for x in g]), ''.join([str(x) for x in g2]))
				genotype_network.add_edge(''.join([str(x) for x in g2]), ''.join([str(x) for x in g]))
			elif pf_map[ph2] > pf_map[ph]:
				genotype_network.add_edge(''.join([str(x) for x in g]), ''.join([str(x) for x in g2]))
			elif pf_map[ph2] < pf_map[ph]:
				genotype_network.add_edge(''.join([str(x) for x in g2]), ''.join([str(x) for x in g]))
	assert len([n for n in genotype_network.nodes()]) == K**L
	assert len([n for n in genotype_network.edges()]) < K**(L) * (K-1) * L * 2
	return genotype_network

def find_peak_identities(phenotype_network, pheno_vs_fitness):
	peaks_identity_list = []
	for ph in phenotype_network:
		if max([pheno_vs_fitness[ph2] for ph2 in phenotype_network.neighbors(ph)]+[-1,]) < pheno_vs_fitness[ph]:
			peaks_identity_list.append(ph)
	return peaks_identity_list


def make_phenotype_network_deleted_nodes(permill_to_keep, GPmap, neighbours_function, phenotype_network_filename=''):
	phenotpes_list = [ph for ph in np.unique(GPmap) if ph > 0]
	nph = len(phenotpes_list)
	if str(permill_to_keep) != 'all':
		phenotpes_list = np.random.choice(phenotpes_list, replace=False, size = int(round(permill_to_keep/1000*nph)))
	if len(phenotpes_list) == 0:
		return nx.Graph()
	phenotype_network = make_phenotype_network(GPmap, neighbours_function, phenotpes_list=phenotpes_list)
	for ph in phenotpes_list:
		if len([n for n in phenotype_network.neighbors(ph)]) == 0:
			phenotype_network.remove_node(ph)
	if phenotype_network_filename:
		phenotype_network_str_nodes = nx.relabel_nodes(phenotype_network, mapping={n: str(n) for n in phenotype_network.nodes()}, copy=True)
		nx.write_gml(phenotype_network_str_nodes, phenotype_network_filename)
	if str(permill_to_keep) != 'all':
		assert int(round(permill_to_keep/1000*nph)) >= len(phenotype_network.nodes())
	for n in phenotype_network:
		assert n in phenotpes_list
	return phenotype_network

def get_directed_network_edges(phenotype_network, pheno_vs_fitness):
	NC_network_directed_edges = []
	for NC1, NC2 in phenotype_network.edges():
		assert pheno_vs_fitness[NC1] != pheno_vs_fitness[NC2]
		if pheno_vs_fitness[NC1] < pheno_vs_fitness[NC2]:
			NC_network_directed_edges.append((NC1, NC2))
		else:
			NC_network_directed_edges.append((NC2, NC1))
	return NC_network_directed_edges

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
	#### test Fibonacci model
	K = 3
	g = (0, 1, 0, 1, 0, 0, 2, 2, 0, 1)
	assert pheno_int_to_str(genotype_to_int_phenotype(g, K), K) == '010100'
	g = (2, 1, 0, 1, 0, 0, 2, 2, 0, 1)
	assert pheno_int_to_str(genotype_to_int_phenotype(g, K), K) == '' and genotype_to_int_phenotype(g, K) == 1
	g = (1, 1, 0, 1, 0, 0, 0, 0, 0, 1)
	assert pheno_int_to_str(genotype_to_int_phenotype(g, K), K) == '' and genotype_to_int_phenotype(g, K) == 0
	K = 4
	g = (0, 1, 0, 1, 0, 0, 2, 2, 0, 1)
	assert pheno_int_to_str(genotype_to_int_phenotype(g, K), K) == '' and genotype_to_int_phenotype(g, K) == 0
	g = (2, 1, 3, 1, 0, 0, 3, 2, 3, 1)
	assert pheno_int_to_str(genotype_to_int_phenotype(g, K), K) == '21' 
	g = (1, 1, 0, 1, 0, 0, 0, 0, 3, 3)
	assert pheno_int_to_str(genotype_to_int_phenotype(g, K), K) == '11010000'
	### mutations by site
	L = len(g)
	for site in range(L):
		assert len(set(neighbours_g_site(g, site, K))) == len(neighbours_g_site(g, site, K)) == (K-1) and len(set([Hamming_dist(g2, g) for g2 in neighbours_g_site(g, site, K)])) == 1 and Hamming_dist(neighbours_g_site(g, site, K)[0], g) == 1
	### low-evolvability null model
	assert len(neighbours_g_not_stopcodon(g, K, L)) == 8 * 2 + 3
	K = 3
	g = (1, 1, 0, 1, 0, 0, 0, 0, 0, 1)
	assert len(neighbours_g_not_stopcodon(g, K, L)) == 10 * 1 
	g = (2, 1, 0, 1, 0, 0, 2, 2, 0, 1)
	assert len(neighbours_g_not_stopcodon(g, K, L)) == 9 * 2
	#########################################################################################################################


