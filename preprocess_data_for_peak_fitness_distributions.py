import numpy as np 
import networkx as nx 
from os.path import isfile
import pandas as pd
import parameters as param
from functions.navigability_functions import  load_dict, choose_non_peak_non_undefined_geno


####################################################################################################################################################
print('data for peak height distribution and outcomes')
####################################################################################################################################################
for distribution in ['uniform', 'exponential']:
	for i, type_structure in enumerate(param.all_types_structue):
		print(type_structure, distribution)
		peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+'_iterations'+str(param.iterations_peaks)+ distribution+ '.csv'
		peak_heights_reached_evolv = './data/peak_heights_reached_evolv_'+type_structure+'_pfmaps'+str(param.no_pf_maps_evolutionary_walks)+'_nruns'+str(param.no_runs_evolutionary_walks)+'pop_size'+str(param.pop_size_evolutionary_walks)+ distribution+'.csv'
		NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
		if isfile(peak_evolv_height_filename) and isfile(peak_heights_reached_evolv) and isfile(NC_evolv_filename):
			NC_vs_evolv = load_dict(NC_evolv_filename)
			df_peak_evolv_height = pd.read_csv(peak_evolv_height_filename)
			df_peak_heights_reached_evolv = pd.read_csv(peak_heights_reached_evolv)
			if distribution == 'uniform':
				xaxis = np.linspace(0, max(df_peak_evolv_height['peak height'].tolist()), num=50) #same across all iterations
			else:
				xaxis = np.power(10, np.linspace(np.log10(min(df_peak_evolv_height['peak height'].tolist())), np.log10(max(df_peak_evolv_height['peak height'].tolist())), num=50)) #same across all iterations
			data = {'G': xaxis}
			
			####
			iteration_vs_peak_cdf, iteration_vs_peak_cdf_kimura = {}, {}
			for iteration in range(param.no_pf_maps_evolutionary_walks):
				peak_heights = df_peak_evolv_height[df_peak_evolv_height['iteration'] == iteration]['peak height'].tolist()
				peak_heights_reached = df_peak_heights_reached_evolv[df_peak_heights_reached_evolv['PF map index'] == iteration]
				if len(peak_heights_reached):
					assert len(set(peak_heights_reached['PF map index'].tolist())) == 1 and peak_heights_reached['PF map index'].tolist()[0] == iteration
					peak_heights_reached_kimura = peak_heights_reached[peak_heights_reached['type_walk'] == 'Kimura']['final fitness'].tolist()
					data['peak cdf - iteration ' + str(iteration)] = [len([x for x in peak_heights if x < h])/len(peak_heights) for h in xaxis]
					data['peak reached cdf - iteration ' + str(iteration)] = [len([x for x in peak_heights_reached_kimura if x < h])/len(peak_heights_reached_kimura) for h in xaxis]
				else:
					data['peak cdf - iteration ' + str(iteration)] = [np.nan]*len(xaxis)
					data['peak reached cdf - iteration ' + str(iteration)] = [np.nan]*len(xaxis)	
					print('no data', type_structure, iteration)
			df = pd.DataFrame.from_dict(data)
			df.to_csv('./data/cdf_peaks_' +'_'+type_structure+'_pfmaps'+str(param.no_pf_maps_evolutionary_walks)+'_nruns'+str(param.no_runs_evolutionary_walks)+'pop_size'+str(param.pop_size_evolutionary_walks)+ distribution+'.csv')
