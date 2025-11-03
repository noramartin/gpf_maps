import numpy as np 
import networkx as nx 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from os.path import isfile
from scipy.stats import gmean, spearmanr, pearsonr
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import parameters as param
import parameters_plots as plotparam
from functions.navigability_functions import isundefinedpheno, load_dict
from collections import Counter
#import pyinstrument 

####################################################################################################################################################
####################################################################################################################################################
distribution = 'uniform'
####################################################################################################################################################
map_vs_exp_no_peaks = {}
####################################################################################################################################################
print('ruggedness plot main text')
####################################################################################################################################################
####################################################################################################################################################

xlims1, xlims2 = [0.01, 0.01], [0.01, 0.01]
f, ax = plt.subplots(ncols  = 2, nrows=2, figsize=(5, 5.5), gridspec_kw={'width_ratios': [1, 0.51]})
for type_structure in param.all_types_structue:
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
	total_peak_size_filename = './data/total_peak_size_'+type_structure+'_iterations'+str(param.iterations_peaks)+distribution+ '.npy'
	total_peak_number_filename = './data/total_peak_number_'+type_structure+'_iterations'+str(param.iterations_peaks)+distribution+ '.npy'
	NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
	if isfile(NC_evolv_filename) and isfile(total_peak_size_filename) and isfile(NC_vs_size_filename) and isfile(total_peak_number_filename):
		NC_vs_evolv = load_dict(NC_evolv_filename)
		print(type_structure, 'minimum-evolvability NC', min([e for e in NC_vs_evolv.values()]))
		total_peak_size_list = np.load(total_peak_size_filename)
		NC_vs_size = load_dict(NC_vs_size_filename)
		predicted_size_peaks = np.sum([NC_vs_size[NC]/(e+1) for NC, e in NC_vs_evolv.items()])
		mean = np.mean(total_peak_size_list)
		min_peak, max_peak = np.percentile(total_peak_size_list, 20), np.percentile(total_peak_size_list,80)
		######
		total_peak_number_list = np.load(total_peak_number_filename)
		mean_number = np.mean(total_peak_number_list)
		min_peak_no, max_peak_no = np.percentile(total_peak_number_list, 20), np.percentile(total_peak_number_list,80)
		predicted_number_peaks = np.sum([1/(e+1) for NC, e in NC_vs_evolv.items()])
		########
		ax[1, 0].errorbar([predicted_size_peaks/K**L,], [mean/K**L, ], yerr=([(mean - min_peak)/K**L,], [(max_peak - mean)/K**L,]), c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')
		ax[1, 0].set_xlabel(r'ruggedness prediction'+'\n'+r'from $|NC|$ '+'&'+ r' $\epsilon_{NC}$ data') 
		ax[1, 0].set_ylabel('mean ruggedness\nin simulations')
		xlims1[0] = min(min(xlims1), min_peak/K**L)
		xlims1[1] = max(max(xlims1), max_peak/K**L)#mean/K**L + std/K**L)

		########
		ax[0, 0].errorbar([predicted_number_peaks/len(NC_vs_evolv),], [mean_number/len(NC_vs_evolv), ], yerr=([(mean_number - min_peak_no)/len(NC_vs_evolv),], [(max_peak_no - mean_number)/len(NC_vs_evolv),]), c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')
		ax[0, 0].set_xlabel(r'# peaks (norm.)'+' prediction'+'\n'+r'from $\epsilon_{NC}$ data') 
		ax[0, 0].set_ylabel('mean # peaks (norm.)\nin simulations')
		xlims2[0] = min(min(xlims2), min_peak_no/len(NC_vs_evolv))
		xlims2[1] = max(max(xlims2), max_peak_no/len(NC_vs_evolv))#mean/K**L + std/K**L)
		map_vs_exp_no_peaks[type_structure] = predicted_number_peaks
	else:
		print('not found', NC_evolv_filename, total_peak_size_filename, NC_vs_size_filename)
for i, xlims in enumerate([xlims2, xlims1]):
	ax[i, 0].set_xscale('log')
	ax[i, 0].set_yscale('log')
	plot_lims = xlims[0]* 0.5, xlims[1]*3
	ax[i, 0].plot(plot_lims, plot_lims, c='k', lw=0.7)#, ls=':')
	ax[i, 0].set_xlim(plot_lims[0], plot_lims[1])
	ax[i, 0].set_ylim(plot_lims[0], plot_lims[1])
	ax[i, 1].axis('off')
ax[0, 1].legend(handles=plotparam.legend_with_errorbar)
[ax[i, 0].annotate('ABCDEFGH'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(2)]
f.tight_layout()
f.savefig('./plots/ruggedness_main_fig'+'_iterations'+str(param.iterations_peaks) +distribution+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

####################################################################################################################################################
####################################################################################################################################################
print('ruggedness plot renormalised')
####################################################################################################################################################
####################################################################################################################################################
type_structure_vs_ruggedness, type_structure_vs_ruggedness_rescaled = {}, {}
for exclude_zero_evolv in [False, True]:
	for rescale_x_viable_genos in [False, True]:
		if exclude_zero_evolv:
			filename_addition_map = 'exclude_zero_evolv'
		else:
			filename_addition_map = ''
		######
		if rescale_x_viable_genos and exclude_zero_evolv:
			axis_label_addition = '\n(out of viable genotypes\nin NCs with '+r'$\epsilon_{NC} \geq 1$)'
		elif rescale_x_viable_genos and not exclude_zero_evolv:
			axis_label_addition = '\n(out of viable genotypes)'
		elif not rescale_x_viable_genos and not exclude_zero_evolv:
			axis_label_addition = ''
		elif exclude_zero_evolv and not rescale_x_viable_genos:
			axis_label_addition = '\n(NCs only counted as peaks\nif '+r'$\epsilon_{NC} \geq 1$)'
		####################################################################################################################################################
		f, ax = plt.subplots(ncols  = 3, figsize=(8, 3), gridspec_kw={'width_ratios': [1, 1] + [0.5]})
		xlims1 = [0.01, 1]
		for type_structure in param.all_types_structue:
			K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]	
			NC_evolv_filename = './data/NC_evolv_'+type_structure+filename_addition_map+ '.csv'
			total_peak_size_filename = './data/total_peak_size_'+type_structure+filename_addition_map+'_iterations'+str(param.iterations_peaks)+distribution+ '.npy'
			NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
			if isfile(NC_evolv_filename) and isfile(total_peak_size_filename) and isfile(NC_vs_size_filename):

				NC_vs_evolv = load_dict(NC_evolv_filename,)
				total_peak_size_list = np.load(total_peak_size_filename)
				NC_vs_size = load_dict(NC_vs_size_filename)
				if exclude_zero_evolv:
					assert min([e for e in NC_vs_evolv.values()]) >= 1
				predicted_size_peaks = np.sum([NC_vs_size[NC]/(e+1) for NC, e in NC_vs_evolv.items()])
				min_peak, max_peak, mean = np.percentile(total_peak_size_list, 20), np.percentile(total_peak_size_list, 80), np.mean(total_peak_size_list)
				if not rescale_x_viable_genos:
					norm = K**L
				else:
					norm = sum([NC_vs_size[NC] for NC in NC_vs_evolv.keys() if NC > 0])
					assert norm <= K**L
				ax[1].errorbar([predicted_size_peaks/norm,], [mean/norm, ], yerr=([(mean - min_peak)/norm,], [(max_peak - mean)/norm,]), c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')
				ax[1].set_xlabel(r'prediction from $|NC|$ '+'&'+ r' $\epsilon_{NC}$:'+ '\nfraction of\ngenotypes that are peaks'+axis_label_addition) 
				ax[1].set_ylabel('simulation: fraction of\ngenotypes that are peaks'+axis_label_addition, fontsize='small')
				xlims1[0] = min(min(xlims1), min_peak/norm)
				xlims1[1] = max(max(xlims1), max_peak/norm)
				if not rescale_x_viable_genos and not exclude_zero_evolv:
					type_structure_vs_ruggedness[type_structure] = (mean/norm, ([(mean - min_peak)/norm,], [(max_peak - mean)/norm,]))
				if rescale_x_viable_genos and not exclude_zero_evolv:
					type_structure_vs_ruggedness_rescaled[type_structure] = (mean/norm, ([(mean - min_peak)/norm,], [(max_peak - mean)/norm,]))
				########
				ax[0].set_ylabel('simulation: fraction of\ngenotypes that are peaks'+axis_label_addition, fontsize='small')	
				if rescale_x_viable_genos and exclude_zero_evolv: 
					ax[0].errorbar([type_structure_vs_ruggedness_rescaled[type_structure][0],], [mean/norm, ], yerr=([(mean - min_peak)/norm,], [(max_peak - mean)/norm,]), xerr=type_structure_vs_ruggedness[type_structure][1],
					            c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')			
					ax[0].set_xlabel('simulation: fraction of\ngenotypes that are peaks\n(out of viable genotypes)') 
					xlims1[0] = min(min(xlims1), type_structure_vs_ruggedness_rescaled[type_structure][0])	
					xlims1[1] = max(max(xlims1), type_structure_vs_ruggedness_rescaled[type_structure][0])				
				else:
					ax[0].errorbar([type_structure_vs_ruggedness[type_structure][0],], [mean/norm, ], yerr=([(mean - min_peak)/norm,], [(max_peak - mean)/norm,]), xerr=type_structure_vs_ruggedness[type_structure][1],
					            c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')			
					ax[0].set_xlabel('simulation: fraction of\ngenotypes that are peaks\n(normalised by all genotypes)') 
					xlims1[0] = min(min(xlims1), type_structure_vs_ruggedness[type_structure][0])	
					xlims1[1] = max(max(xlims1), type_structure_vs_ruggedness[type_structure][0])	

		for i in range(2):
			plot_lims = xlims1[0]* 0.5, xlims1[1]*3
			ax[i].plot(plot_lims, plot_lims, c='k', lw=0.7)#, ls=':')
			ax[i].set_xlim(plot_lims[0], plot_lims[1])
			ax[i].set_ylim(plot_lims[0], plot_lims[1])
			ax[i].set_xscale('log')
			ax[i].set_yscale('log')
		ax[-1].legend(handles=plotparam.legend_with_errorbar)
		ax[-1].axis('off')
		f.tight_layout()
		if rescale_x_viable_genos:
			f.savefig('./plots/ruggedness_renorm'+filename_addition_map+'rescale_x_viable_genos'+'_iterations'+str(param.iterations_peaks) +distribution+'.png', dpi=300, bbox_inches='tight')
		else:
			f.savefig('./plots/ruggedness_renorm'+filename_addition_map+'_iterations'+str(param.iterations_peaks) +distribution+'.png', dpi=300, bbox_inches='tight')
		plt.close('all')

####################################################################################################################################################
print('reduced dimensionality and correlations')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 3, figsize=(10, 3), gridspec_kw={'width_ratios': [0.7, 1, 0.5]})
for plotindex, plot_type in enumerate(['correlation', 'dimension']):
	ylims = [0.01, 0.01]
	for j, type_structure in enumerate([s for s in param.all_types_structue if 'null' not in s]):
		K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
		list_dim, list_size_peaks, list_size_peaks_predicted = [], [], []
		if plot_type == 'dimension':
			range_values = range(1, L + 1)
		else:
			range_values = ['none', 'all']
		for ind_corr, dimensionality in enumerate(range_values):
			if (plot_type == 'dimension' and dimensionality == L) or (plot_type == 'correlation' and dimensionality == 'none'):
				type_structure_dim = type_structure # unperturbed map
			elif plot_type == 'dimension':
				type_structure_dim = type_structure + 'dim' + str(dimensionality)	
			else:
				type_structure_dim = type_structure + 'permuted'	
			if (plot_type == 'dimension' and dimensionality == L) or (plot_type == 'correlation' and dimensionality == 'none'):
				NC_evolv_filename = './data/NC_evolv_'+type_structure_dim+ '_summary.csv'
			else:
				NC_evolv_filename = './data/NC_evolv_'+type_structure_dim+ '.csv'
			total_peak_size_filename = './data/total_peak_size_'+type_structure_dim+'_iterations'+str(param.iterations_peaks)+distribution+ '.npy'
			if  isfile(NC_evolv_filename) and isfile(total_peak_size_filename):
				df_evolv = pd.read_csv(NC_evolv_filename)
				list_size_peaks_predicted.append(np.sum([s/(e+1) for s, e in zip(df_evolv['size'].tolist(), df_evolv['evolv'].tolist())])/K**L)
				total_peak_size_list = np.load(total_peak_size_filename)
				list_size_peaks.append(np.mean(total_peak_size_list)/K**L)
				if plot_type == 'dimension':
					list_dim.append(dimensionality/L)
				else:
					list_dim.append(ind_corr + 0.05* (j - 0.5* len(type_structure))/len(type_structure))
			elif 'null' not in type_structure:
				print('not found', NC_evolv_filename, total_peak_size_filename)
		ylims = [min(ylims + list_size_peaks), max(ylims + list_size_peaks)]
		#############
		ax[plotindex].scatter(list_dim, list_size_peaks, marker='o', color=plotparam.type_structure_vs_color[type_structure], s=8)
		ax[plotindex].plot(list_dim, list_size_peaks_predicted, color=plotparam.type_structure_vs_color[type_structure], lw=0.8, ls=':')
		if plot_type == 'correlation':
			ax[plotindex].set_xticks([0, 1])
			ax[plotindex].set_xlim(-0.2, 1.2)
			ax[plotindex].set_xticklabels(['GP map', 'HoC-GP map\n(correlations removed)'], rotation=30, ha='right')
		ax[plotindex].set_yscale('log')
		ax[plotindex].set_ylim(min(ylims) * 0.3, 3* max(ylims))
		ax[plotindex].set_ylabel('fraction of genotypes\nthat are peaks')
		if plot_type == 'dimension':
			ax[plotindex].set_xlabel('normalised dimensionality\n(fraction of sites allowed to mutate)')
[ax[i].annotate('ABCDEFGH'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(2)]
ax[-1].legend(handles=plotparam.legend_without_Fibonaccinull)
ax[-1].axis('off')
f.tight_layout()
f.savefig('./plots/number_peaks_evolv'+'_iterations'+str(param.iterations_peaks) +distribution+'.png', bbox_inches='tight', dpi=300)

####################################################################################################################################################
print('plot size of peaks with analytic - Fibonacci')
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 2, figsize=(5.3, 2.5), width_ratios = [1, 1.1])
xlims1 = [0.01, 0.01]
for type_structure in ['Fibonacci_12_3', 'Fibonacci_null_12_3']:
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	######## analytics
	length_coding_region = np.arange(0, L)
	size = [(K ** (L - l - 1)) for l in length_coding_region]
	number_pheno = [(K-1)**l for l in length_coding_region]
	if 'null' not in type_structure:
		evolv = [(K - 1) * l + (K - 1) * sum([(K-1)**lnc for lnc in range(0, L - l - 1 )]) for l in length_coding_region] #new coding region can be up to L -l -2
	else:
		evolv = [(K - 2) * l for l in length_coding_region]
	predicted_size_peaks = sum([n*f/(e+1) for f, n, e in zip(size, number_pheno, evolv)])
	######
	#############
	total_peak_size_filename = './data/total_peak_size_'+type_structure+'_iterations'+str(param.iterations_peaks)+distribution+ '.npy'
	if  isfile(total_peak_size_filename):
		total_peak_size_list = np.load(total_peak_size_filename)
		min_peak, max_peak, mean = np.percentile(total_peak_size_list, 20), np.percentile(total_peak_size_list, 80), np.mean(total_peak_size_list)
		ax[0].errorbar([predicted_size_peaks/K**L,], [mean/K**L, ], yerr=([(mean - min_peak)/norm,], [(max_peak - mean)/norm,]), c=plotparam.type_structure_vs_color[type_structure], ms=5, marker='o')
		ax[0].set_xlabel('analytic prediction:\nfraction of genotypes\nthat are peaks')
		ax[0].set_ylabel('simulation\nresult: fraction\nof genotypes\nthat are peaks')
		xlims1[0] = min(min(xlims1), min([min_peak/K**L, predicted_size_peaks/K**L]))
		xlims1[1] = max(max(xlims1), max([max_peak/K**L, predicted_size_peaks/K**L]))
ax[0].set_xscale('log')
ax[0].set_yscale('log')
plot_lims = xlims1[0]* 0.5, xlims1[1]*3
ax[0].plot(plot_lims, plot_lims, c='k', linestyle=':')
ax[0].set_xlim(plot_lims[0], plot_lims[1])
ax[0].set_ylim(plot_lims[0], plot_lims[1]) 
custom_lines = [Line2D([0], [0], mfc=plotparam.type_structure_vs_color[type_structure], linestyle='', marker='o', label=plotparam.type_structure_vs_label[type_structure], mew=0, ms=5) for type_structure in ['Fibonacci_12_3', 'Fibonacci_null_12_3']]+ [Line2D([0], [0], c='grey', lw=0.5, ls='-', marker=None, label=r'$20^{th}$-$80^{th}$ percent.', mew=0, ms=5)]
ax[1].legend(handles=custom_lines)
ax[1].axis('off')
f.tight_layout()
f.savefig('./plots/fibonacci_analytic_peaks_L'+str(L)+'_'+str(K)+'_iterations'+str(param.iterations_peaks) +'.png', bbox_inches='tight', dpi=250)

###################################################################################################################################################
print('plot peak height vs evolv - single plot ')
####################################################################################################################################################
f, ax = plt.subplots(figsize=(5, 2.5), ncols = 2)
plot_type = 'evolvability'
for col, (distribution, type_structure) in enumerate([('uniform', 'RNA_12'), ('exponential', 'RNA_12')]):
	peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+'_iterations'+str(param.iterations_peaks)+ distribution+ '.csv'
	print(peak_evolv_height_filename)
	if isfile(peak_evolv_height_filename):
		df_peak_evolv_height = pd.read_csv(peak_evolv_height_filename)
		assert 0.9 <= len(df_peak_evolv_height)/(map_vs_exp_no_peaks[type_structure]*param.iterations_peaks) <= 1.1 #check number of peaks in file as expected
		sns.lineplot(data=df_peak_evolv_height, x='peak ' +plot_type, y='peak height', color = 'k', estimator='mean', errorbar='sd', ax = ax[col], err_kws={'alpha': 0.4})
		x_values = [x for x in range(int(min([x for x in df_peak_evolv_height['peak ' +plot_type].tolist() if x > 0])), int(max(df_peak_evolv_height['peak ' +plot_type].tolist() ))+ 5)]
		if distribution == 'uniform':
			predicted_mean = np.divide(np.array(x_values) + 1, np.array(x_values) + 2)
			predicted_std =  np.power(np.divide(np.array(x_values) + 1, np.array(x_values) + 3) - np.power(np.divide(np.array(x_values) + 1, np.array(x_values) + 2), 2), 0.5)#\frac{\epsilon_p +1}{\epsilon_p +3} - \frac{(\epsilon_p +1)^2}{(\epsilon_p +2)^2}
		elif distribution == 'exponential':
			predicted_mean = [sum([1/n for n in range(1, x+2)]) for x in x_values] #sum up to epsilon + 1
			predicted_std = np.power([sum([1/n**2 for n in range(1, int(x)+2)]) for x in x_values], 0.5)		
		lower_estimate = predicted_mean - predicted_std
		upper_estimate = predicted_mean + predicted_std
		ax[col].plot(x_values, predicted_mean, c='r', linestyle=(0, (3, 7)))
		ax[col].plot(x_values, lower_estimate, c='r', linestyle=(0, (3, 7)))
		ax[col].plot(x_values, upper_estimate, c='r', linestyle=(0, (3, 7)))
		ax[col].set_xlabel(r'NC evolvability $\epsilon_{NC}$')
		ax[col].set_xscale('log')
		ax[col].set_title(plotparam.type_structure_vs_label[type_structure] + ',\n' + distribution + ' PF-map')
		if distribution == 'uniform':
			ax[col].set_ylim(0, 1.03)
	custom_lines = [Line2D([0], [0], mfc=['k', 'r'][i], ls='', marker='o', label=['simulation', 'prediction'][i], mew=0, ms=5) for i in range(2)]
	ax[0].legend(handles=custom_lines, loc=4, frameon=False)
[ax[i].annotate('ABCDEFGH'[i], xy=(0.0, 1.2), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(2)]
f.tight_layout()
f.savefig('./plots/peak_height_evolvability_summary' +'_iterations'+str(param.iterations_peaks)+'.png', bbox_inches='tight', dpi=200)

###################################################################################################################################################
print('plot peak height distribution with evolutionary outcomes - single plot ')
####################################################################################################################################################
type_structure = 'RNA_12'
f, ax = plt.subplots(figsize=(4.6, 5), ncols = 2, nrows=2, width_ratios = [1, 0.6])
for i, distribution in enumerate(['uniform', 'exponential']):
	cdf_file = pd.read_csv('./data/cdf_peaks_' +'_'+type_structure+'_pfmaps'+str(param.no_pf_maps_evolutionary_walks)+'_nruns'+str(param.no_runs_evolutionary_walks)+'pop_size'+str(param.pop_size_evolutionary_walks)+ distribution+'.csv' )
	NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
	if isfile(NC_evolv_filename):
		NC_vs_evolv = load_dict(NC_evolv_filename)
		xaxis = cdf_file['G'].tolist()
		####
		predicted_number_peaks = np.sum([1/(e+1) for NC, e in NC_vs_evolv.items()])
		if distribution == 'uniform':
			cumulative_peak_heights_expected = [sum([h**(e+1)/(e+1) for e in NC_vs_evolv.values()])/predicted_number_peaks for h in xaxis]
		elif distribution == 'exponential':
			cumulative_peak_heights_expected = [sum([(1-np.exp(-1*h))**(e+1)/(e+1) for e in NC_vs_evolv.values()])/predicted_number_peaks for h in xaxis]
		iteration_vs_peak_cdf, iteration_vs_peak_cdf_kimura = {}, {}
		for iteration in range(param.no_pf_maps_evolutionary_walks):
			if not np.isnan(cdf_file['peak cdf - iteration ' + str(iteration)].tolist()[0]):
				iteration_vs_peak_cdf[iteration] = cdf_file['peak cdf - iteration ' + str(iteration)].tolist()
				iteration_vs_peak_cdf_kimura[iteration] = cdf_file['peak reached cdf - iteration ' + str(iteration)].tolist()
		for data, color, symbol in ((iteration_vs_peak_cdf, 'k', '-.'), (iteration_vs_peak_cdf_kimura, 'g', '--')):
			mean = [np.mean([data[iteration][j] for iteration in iteration_vs_peak_cdf]) for j in range(len(xaxis))]
			std = [np.std([data[iteration][j] for iteration in iteration_vs_peak_cdf]) for j in range(len(xaxis))]
			ax[i, 0].errorbar(xaxis, mean, yerr=std, c=color, zorder=4, alpha=0.6)
		ax[i, 0].plot(xaxis, cumulative_peak_heights_expected, c='r', zorder=5, ls=':')
		ax[i, 0].set_xlabel(r'fitness $G$')
		ax[i, 0].set_ylabel(r'$P($ peak height $ < G)$')
		ax[i, 0].set_title(plotparam.type_structure_vs_label[type_structure]+ ',\n' + distribution + ' PF-map')
		ax[i, 1].axis('off')
	if distribution == 'exponential':
		ax[i, 0].set_xscale('log')
custom_lines = [Line2D([0], [0], mfc=['grey', 'r', 'g'][i], ls='', marker='o', label=['all peaks\n(simulation)', 'all peaks\n(prediction)', 'adaptive walk\n(simulation)'][i], mew=0, ms=5) for i in range(3)]
ax[0, 1].legend(handles=custom_lines, frameon=False)#fontsize='small')
[ax[i, 0].annotate('ABCDEFGH'[i], xy=(0.0, 1.2), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(2)]
f.tight_layout()
f.savefig('./plots/peak_height_evolvability_pop_iterations'+str(param.iterations_peaks)+'.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('plot peak height distribution and outcomes')
####################################################################################################################################################
for distribution in ['exponential', 'uniform']:
	f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 4.75))
	for i, type_structure in enumerate(param.all_types_structue):
		print(type_structure, distribution)
		NC_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
		peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+'_iterations'+str(param.iterations_peaks)+ distribution+ '.csv'
		peak_heights_reached_evolv = './data/peak_heights_reached_evolv_'+type_structure+'_pfmaps'+str(param.no_pf_maps_evolutionary_walks)+'_nruns'+str(param.no_runs_evolutionary_walks)+'pop_size'+str(param.pop_size_evolutionary_walks)+ distribution+'.csv'
		if isfile(peak_evolv_height_filename) and isfile(peak_heights_reached_evolv):
			NC_vs_size = load_dict(NC_size_filename)
			df_peak_evolv_height = pd.read_csv(peak_evolv_height_filename)
			df_peak_heights_reached_evolv = pd.read_csv(peak_heights_reached_evolv)
			if map_vs_exp_no_peaks[type_structure]*param.iterations_peaks < 5*10**6:
				assert 0.9 <= len(df_peak_evolv_height)/(map_vs_exp_no_peaks[type_structure]*param.iterations_peaks) <= 1.1 #check number of peaks in file as expected
			else:
				assert 0.9 <= len(df_peak_evolv_height)/(map_vs_exp_no_peaks[type_structure]*param.no_pf_maps_evolutionary_walks) <= 1.1 #check number of peaks in file as expected
			binsx = np.linspace(min([np.log10(NC_vs_size[NC]) for NC in df_peak_evolv_height['peak NC'].tolist()]), max([np.log10(NC_vs_size[NC]) for NC in df_peak_evolv_height['peak NC'].tolist()]), 10)
			binsy = np.linspace(min(df_peak_evolv_height['peak height'].tolist()), max(df_peak_evolv_height['peak height'].tolist()), 10)
			hist_dynamics_list, hist_all_list = [], []
			for iteration in range(param.no_pf_maps_evolutionary_walks):
				#####
				peaks_iteration_df = df_peak_evolv_height[df_peak_evolv_height['iteration'] == iteration]
				assert len(set(peaks_iteration_df['iteration'].tolist())) <= 1 and (len(peaks_iteration_df) == 0 or peaks_iteration_df['iteration'].tolist()[0] == iteration)
				peak_NCs = peaks_iteration_df['peak NC'].tolist()
				peak_heights = peaks_iteration_df['peak height'].tolist()
				peaks_reached_df = df_peak_heights_reached_evolv[df_peak_heights_reached_evolv['PF map index'] == iteration]
				assert len(set(peaks_reached_df['PF map index'].tolist())) <= 1 and (len(set(peaks_reached_df['PF map index'].tolist())) == 0 or peaks_reached_df['PF map index'].tolist()[0] == iteration)
				if len(set(peaks_reached_df['PF map index'].tolist())) == 1:
					####### need to check that peak_NCs vs peak_heights matches the evolutionary info
					NC_vs_height = {NC: h for NC, h in zip(peak_NCs, peak_heights)}
					for rowi, row in peaks_reached_df.iterrows():
						assert abs(NC_vs_height[row['final NC']] - row['final fitness']) < 0.001
					########
					NCs_reached = peaks_reached_df[peaks_reached_df['type_walk'] == 'Kimura']['final NC'].tolist()
					list_peak_reached_fitness = [NC_vs_height[NC] for NC in NCs_reached]				
					hist_dynamics_list.append(np.copy( np.divide(np.histogram2d([np.log10(NC_vs_size[NC]) for NC in NCs_reached], list_peak_reached_fitness, bins=(binsx, binsy))[0], len(list_peak_reached_fitness))))
					hist_all_list.append(np.copy(np.divide(np.histogram2d([np.log10(NC_vs_size[NC]) for NC in peak_NCs], peak_heights, bins=(binsx, binsy))[0], len(peak_heights))))
			####
			histcounts_all, histcounts_dynamics = np.zeros(hist_all_list[0].shape), np.zeros(hist_all_list[0].shape)
			for index, z in np.ndenumerate(histcounts_all):
				if len(hist_all_list) > 1:
					histcounts_all[index] = np.mean([m[index] for m in hist_all_list])
					histcounts_dynamics[index] = np.mean([m[index] for m in hist_dynamics_list])
			max_abs_bincount = np.max(np.abs(histcounts_all - histcounts_dynamics))
			cm = ax[i//5, i%5].pcolormesh(binsx, binsy, (histcounts_dynamics - histcounts_all).T, vmin=-1*max_abs_bincount, vmax = max_abs_bincount, cmap=plt.cm.PRGn)
			cb = f.colorbar(cm, ax=ax[i//5, i%5])
			cb.set_label('normalised count:\nrealised - all') 
			ax[i//5, i%5].set_xlabel(r'$\log_{10} |NC|$')
			ax[i//5, i%5].set_ylabel('peak height')
			#ax[i//5, i%5].set_xscale('log')
			ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure], fontsize=12 )
		ax[9//5, 9%5].axis('off')
	f.tight_layout()
	f.savefig('./plots/peak_height_outcomes' +'_iterations'+str(param.iterations_peaks)+ distribution+'.png', bbox_inches='tight', dpi=200)
	plt.close('all')

####################################################################################################################################################
print('plot peak height distribution and outcomes')
####################################################################################################################################################
plot_type_list = ['iteration_'+str(i) for i in range(param.no_pf_maps_evolutionary_walks)] +['mean'] 
plot_all = False

for distribution in ['uniform', 'exponential']:
	difference_list, lower_cdf_list, higher_cdf_list, equal_cdf_list = {type_structure: [] for type_structure in param.all_types_structue}, {type_structure: [] for type_structure in param.all_types_structue}, {type_structure: [] for type_structure in param.all_types_structue}, {type_structure: [] for type_structure in param.all_types_structue}
	type_structure_vs_cdf_filename = {type_structure: './data/cdf_peaks_' +'_'+type_structure+'_pfmaps'+str(param.no_pf_maps_evolutionary_walks)+'_nruns'+str(param.no_runs_evolutionary_walks)+'pop_size'+str(param.pop_size_evolutionary_walks)+ distribution+'.csv' for type_structure in param.all_types_structue}
	type_structure_vs_cdf_data = {type_structure: pd.read_csv(filename) for type_structure, filename in type_structure_vs_cdf_filename.items() if isfile(filename)}
	for plot_type in plot_type_list:
		if plot_all or plot_type.startswith('mean'):
			f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(17, 4.75))
		for i, type_structure in enumerate(param.all_types_structue):
			NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
			if isfile(NC_evolv_filename):
				NC_vs_evolv = load_dict(NC_evolv_filename)
				xaxis = type_structure_vs_cdf_data[type_structure]['G'].tolist()
				####
				predicted_number_peaks = np.sum([1/(e+1) for NC, e in NC_vs_evolv.items()])
				evolv_list = np.array([e for e in NC_vs_evolv.values()])
				if distribution == 'uniform':
					cumulative_peak_heights_expected = [np.sum(np.divide(np.power(h, evolv_list + 1), evolv_list+1))/predicted_number_peaks for h in xaxis] #[sum([h**(e+1)/(e+1) for e in NC_vs_evolv.values()])/predicted_number_peaks for h in xaxis]
				elif distribution == 'exponential': 
					cumulative_peak_heights_expected = [np.sum(np.divide(np.power(1-np.exp(-1*h), evolv_list + 1), evolv_list+1))/predicted_number_peaks for h in xaxis]
				####
				iteration_vs_peak_cdf, iteration_vs_peak_cdf_kimura, iterations_with_data = {}, {}, []
				if plot_type == 'mean':
					iteration_list = np.arange(param.no_pf_maps_evolutionary_walks)
				else:
					iteration_list = [int(plot_type.split('_')[1]),]
				for iteration in iteration_list:
					if type_structure in type_structure_vs_cdf_data and not np.isnan(type_structure_vs_cdf_data[type_structure]['peak cdf - iteration ' + str(iteration)].tolist()[0]):
						iteration_vs_peak_cdf[iteration] = type_structure_vs_cdf_data[type_structure]['peak cdf - iteration ' + str(iteration)].tolist()
						iteration_vs_peak_cdf_kimura[iteration] = type_structure_vs_cdf_data[type_structure]['peak reached cdf - iteration ' + str(iteration)].tolist()
						iterations_with_data.append(iteration)
				for data, color, symbol in ((iteration_vs_peak_cdf, 'k', '-.'), (iteration_vs_peak_cdf_kimura, 'g', '--')):
					if plot_type == 'mean':
						
						mean = [np.mean([data[iteration][j] for iteration in iterations_with_data]) for j in range(len(xaxis))]
						std = [np.std([data[iteration][j] for iteration in iterations_with_data]) for j in range(len(xaxis))]
						ax[i//5, i%5].errorbar(xaxis, mean, yerr=std, c=color, zorder=4, alpha=0.6)
					elif plot_type.startswith('iteration_') and plot_all and int(plot_type.split('_')[1]) in data:
						ax[i//5, i%5].plot(xaxis, data[int(plot_type.split('_')[1])], c=color, ls=symbol)
				if plot_type.startswith('iteration_') and int(plot_type.split('_')[1]) in iteration_vs_peak_cdf_kimura:
					iteration = int(plot_type.split('_')[1])
					difference_list[type_structure].append(np.mean([y - x for y, x in zip(iteration_vs_peak_cdf[iteration], iteration_vs_peak_cdf_kimura[iteration])]))
					all_higher = [y for y, x in zip(iteration_vs_peak_cdf[iteration], iteration_vs_peak_cdf_kimura[iteration]) if y > x + 0.01]
					lower_cdf_list[type_structure].append(len([y for y, x in zip(iteration_vs_peak_cdf[iteration], iteration_vs_peak_cdf_kimura[iteration]) if x > y + 0.01]))
					higher_cdf_list[type_structure].append(len([y for y, x in zip(iteration_vs_peak_cdf[iteration], iteration_vs_peak_cdf_kimura[iteration]) if y > x + 0.01]))
					equal_cdf_list[type_structure].append(len([y for y, x in zip(iteration_vs_peak_cdf[iteration], iteration_vs_peak_cdf_kimura[iteration]) if abs(y - x) > 0.01]))
				if plot_all or plot_type.startswith('mean'):
					print(type_structure, 'iterations_with_data', len(iterations_with_data)/len(iteration_list))
					ax[i//5, i%5].plot(xaxis, cumulative_peak_heights_expected, c='r', zorder=5, ls=':')
					ax[i//5, i%5].set_xlabel(r'fitness $G$')
					ax[i//5, i%5].set_ylabel(r'$P($ peak height $ < G)$')
					ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure] )
			if plot_all or plot_type.startswith('mean'):
				if distribution == 'exponential':
					ax[i//5, i%5].set_xscale('log')
				[ax[i//5, i%5].annotate('ABCDEFGHIJKLMN'[i], xy=(0.01, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(9)]
				ax[9//5, 9%5].axis('off')
		if plot_all or plot_type.startswith('mean'):
			f.tight_layout()
			f.savefig('./plots/peak_height_dist'+plot_type+'_iterations'+str(param.iterations_peaks)+ distribution+'.png', bbox_inches='tight', dpi=200)
			plt.close('all')
	f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(17, 6))
	for i, type_structure in enumerate(param.all_types_structue):
		ax[i//5, i%5].hist(difference_list[type_structure], color='b', bins=40)
		ax[i//5, i%5].set_xlabel('mean difference:\nall peaks CDF - walks CDF')
		fraction_all_higher = len([x for x in lower_cdf_list[type_structure] if x == 0])/len(lower_cdf_list[type_structure])
		fraction_dynamic_higher = len([x for x in higher_cdf_list[type_structure] if x == 0])/len(higher_cdf_list[type_structure])
		fraction_equal = len([x for x in equal_cdf_list[type_structure] if x == 0])/len(equal_cdf_list[type_structure])
		fraction_pos = len([x for x in difference_list[type_structure] if x > 0])/len(difference_list[type_structure])
		ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure]+'\nall peaks CDF + 0.01 > walks CDF: '+str(int(round(fraction_all_higher*100)))+'%'+'\nwalks CDF + 0.01 > all peaks CDF: '+str(int(round(fraction_dynamic_higher*100)))+'%' +'\nCDFs equal: '+str(int(round(fraction_equal*100)))+'%' + '\nmean difference > 0: ' +str(int(round(fraction_pos*100)))+'%', fontsize=10 )
	f.tight_layout()
	f.savefig('./plots/difference_peak_height_dist'+'_iterations'+str(param.iterations_peaks)+ distribution+'.png', bbox_inches='tight', dpi=200)


####################################################################################################################################################
print('plot peak height vs evolv')
####################################################################################################################################################
for plot_type in ['evolvability', 'size', 'evolvability_manual']:
	for distribution in ['exponential', 'uniform']:
		print('\n', plot_type, distribution)
		f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 4.75))
		for i, type_structure in enumerate(param.all_types_structue):
			peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+'_iterations'+str(param.iterations_peaks)+ distribution+ '.csv'
			if isfile(peak_evolv_height_filename):
				df_peak_evolv_height = pd.read_csv(peak_evolv_height_filename, nrows=5*10**6)
				if map_vs_exp_no_peaks[type_structure]*param.iterations_peaks < 5*10**6:
					assert 0.9 <= len(df_peak_evolv_height)/(map_vs_exp_no_peaks[type_structure]*param.iterations_peaks) <= 1.1 #check number of peaks in file as expected
				else:
					assert 0.9 <= len(df_peak_evolv_height)/(5*10**6) <= 1.1 #check number of peaks in file as expected
				if len(df_peak_evolv_height['peak height']) > 10:					
					if 'null' not in type_structure and 'manual' not in plot_type:
						print(type_structure, 'correlation peak height', plot_type, spearmanr(df_peak_evolv_height['peak height'].tolist(), df_peak_evolv_height['peak '+plot_type].tolist()))
						print(type_structure, 'Pearson correlation peak height', plot_type, pearsonr(df_peak_evolv_height['peak height'].tolist(), np.log10(df_peak_evolv_height['peak '+plot_type].tolist())))
						if 'Fibonacci' in type_structure and plot_type == 'size': #rapid increase of evolv only at 10^2
							height_list_largeNCs, size_list_largeNCs = zip(*[(x, y) for x, y in zip(df_peak_evolv_height['peak height'].tolist(), df_peak_evolv_height['peak '+plot_type].tolist()) if y > 100] )
							print(type_structure, 'large NCs correlation peak height', plot_type, spearmanr(height_list_largeNCs, size_list_largeNCs))
							print(type_structure, 'large NCs Pearson correlation peak height', plot_type, pearsonr(height_list_largeNCs, np.log10(size_list_largeNCs)))
					if plot_type == 'evolvability':
						sns.lineplot(data=df_peak_evolv_height, x='peak '+plot_type, y='peak height', color = 'k', estimator='mean', errorbar='sd', ax = ax[i//5, i%5], err_kws={'alpha': 0.4})
					if plot_type.startswith('evolvability'):
						x_values = [x for x in range(int(min([x for x in df_peak_evolv_height['peak evolvability'].tolist() if x > 0])), int(max(df_peak_evolv_height['peak evolvability'].tolist() ))+ 5)]
						if distribution == 'uniform':
							predicted_mean = np.divide(np.array(x_values) + 1, np.array(x_values) + 2)
							predicted_std =  np.power(np.divide(np.array(x_values) + 1, np.array(x_values) + 3) - np.power(np.divide(np.array(x_values) + 1, np.array(x_values) + 2), 2), 0.5)#\frac{\epsilon_p +1}{\epsilon_p +3} - \frac{(\epsilon_p +1)^2}{(\epsilon_p +2)^2}
						elif distribution == 'exponential':
							predicted_mean = [sum([1/n for n in range(1, x+2)]) for x in x_values] #sum up to epsilon + 1
							predicted_std = np.power([sum([1/n**2 for n in range(1, int(x)+2)]) for x in x_values], 0.5)
						lower_estimate = predicted_mean - predicted_std
						upper_estimate = predicted_mean + predicted_std
						ax[i//5, i%5].plot(x_values, predicted_mean, c='r', linestyle=(0, (3, 7)))
						ax[i//5, i%5].plot(x_values, lower_estimate, c='g', linestyle=(0, (3, 7)))
						ax[i//5, i%5].plot(x_values, upper_estimate, c='g', linestyle=(0, (3, 7)))
						ax[i//5, i%5].set_xlabel(r'NC evolvability $\epsilon_{NC}$')
					if plot_type == 'evolvability_manual':
						mine, maxe = max(0.5, min(df_peak_evolv_height['peak evolvability'].tolist())), max(df_peak_evolv_height['peak evolvability'].tolist()) + 0.01
						bins = [0,] + list(np.power(10, np.linspace(np.log10(mine), np.log10(maxe), num=40)))
						bin_means = [0.5*(bins[i] + bins[i-1]) for i in range(1, len(bins))]
						peak_heights_in_bin = [[] for i in range(len(bin_means))]
						for e, h in zip(df_peak_evolv_height['peak evolvability'].tolist(), df_peak_evolv_height['peak height'].tolist()):
							j = max([i for i, b in enumerate(bins) if b <= e])
							peak_heights_in_bin[j].append(h)
						ax[i//5, i%5].errorbar(bin_means, [np.mean(l) for l in peak_heights_in_bin], yerr=[np.std(l) for l in peak_heights_in_bin], color='grey')
					elif plot_type == 'size':
						ax[i//5, i%5].scatter(df_peak_evolv_height['peak '+plot_type].tolist(), df_peak_evolv_height['peak height'].tolist(), s=1, alpha=0.05, c='r')
						ax[i//5, i%5].set_ylabel('peak height')
						x_size = sorted(list(set([N for N in df_peak_evolv_height['peak '+plot_type].tolist()])))
						mean_fitness = [np.mean(df_peak_evolv_height[df_peak_evolv_height['peak '+plot_type] == N]['peak height'].tolist()) for N in x_size]
						ax[i//5, i%5].plot(x_size, mean_fitness, c='k')
						#if distribution == 'exponential':
						#	ax[i//5, i%5].set_yscale('log')
					if plot_type == 'size':
						ax[i//5, i%5].set_xlabel(r'NC size $|NC|$')
					if not (type_structure == 's_2_8' and plot_type.startswith('evolvability')):
			   			ax[i//5, i%5].set_xscale('log')
				ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])
				if distribution == 'uniform' and plot_type == 'evolvability':
					ax[i//5, i%5].set_ylim(0, 1.03)
			ax[9//5, 9%5].axis('off')
		f.tight_layout()
		f.savefig('./plots/peak_height_'+plot_type+'_iterations'+str(param.iterations_peaks)+ distribution+'.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('plot size of ten highest and lowest peaks')
####################################################################################################################################################
for distribution in ['exponential', 'uniform']:
	print('\n', distribution)
	f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(12, 4.75))
	for i, type_structure in enumerate(param.all_types_structue):
		peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+'_iterations'+str(param.iterations_peaks)+ distribution+ '.csv'
		if isfile(peak_evolv_height_filename):
			df_peak_evolv_height = pd.read_csv(peak_evolv_height_filename)
			mean_size_all, mean_size_bottom = [], []
			for iteration in range(param.no_pf_maps_evolutionary_walks):
				df_peak_evolv_height_it = df_peak_evolv_height[df_peak_evolv_height['iteration'] == iteration]
				if len(df_peak_evolv_height_it['peak size'].tolist()) > 20:
					peak_size_and_height = [(s, h) for s, h in zip(df_peak_evolv_height_it['peak size'].tolist(), df_peak_evolv_height_it['peak height'].tolist())]
					sorted_peak_size = [s[0] for s in sorted(peak_size_and_height, key=lambda x: x[1])]
					quartile_no = len(sorted_peak_size)//5
					mean_size_bottom.append(np.mean(sorted_peak_size[:quartile_no]))
					mean_size_all.append(np.mean(sorted_peak_size))
					assert len(sorted_peak_size) == len(df_peak_evolv_height_it)
			ax[i//5, i%5].scatter(mean_size_bottom, mean_size_all, s=3, c='r')
			ax[i//5, i%5].set_xlabel('mean NC size\nlowest 20% of peaks')
			ax[i//5, i%5].set_ylabel('mean NC size\nall peaks')
			print(distribution, type_structure, 'fraction mean all peaks larger than lowest', len([s for s, t in zip(mean_size_bottom, mean_size_all) if t>s])/len(mean_size_bottom))
			xlims = (min(mean_size_bottom+mean_size_all)*0.5, max(mean_size_bottom+mean_size_all)*2)
			if max(xlims) > 10:
				ax[i//5, i%5].set_xscale('log')
				ax[i//5, i%5].set_yscale('log')
			ax[i//5, i%5].set_xlim(xlims[0], xlims[1])
			ax[i//5, i%5].set_ylim(xlims[0], xlims[1])
			ax[i//5, i%5].plot(xlims, xlims, c='k')
			ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])
		ax[9//5, 9%5].axis('off')
	f.tight_layout()
	f.savefig('./plots/peak_height_'+'size'+'_iterations'+str(param.iterations_peaks)+ distribution+'_2.png', bbox_inches='tight', dpi=200)


###################################################################################################################################################
print('plot peak height vs evolv - single plot ')
####################################################################################################################################################
f, ax = plt.subplots(figsize=(5, 2.5), ncols = 2)
plot_type = 'evolvability'
for col, (distribution, type_structure) in enumerate([('uniform', 'RNA_12'), ('exponential', 'RNA_12')]):
	peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+'_iterations'+str(param.iterations_peaks)+ distribution+ '.csv'
	print(peak_evolv_height_filename)
	if isfile(peak_evolv_height_filename):
		df_peak_evolv_height = pd.read_csv(peak_evolv_height_filename)
		print('number of peaks in dataset, ', len(df_peak_evolv_height))
		sns.lineplot(data=df_peak_evolv_height, x='peak ' +plot_type, y='peak height', color = 'k', estimator='mean', errorbar='sd', ax = ax[col], err_kws={'alpha': 0.4})
		x_values = [x for x in range(int(min([x for x in df_peak_evolv_height['peak ' +plot_type].tolist() if x > 0])), int(max(df_peak_evolv_height['peak ' +plot_type].tolist() ))+ 5)]
		if distribution == 'uniform':
			predicted_mean = np.divide(np.array(x_values) + 1, np.array(x_values) + 2)
			predicted_std =  np.power(np.divide(np.array(x_values) + 1, np.array(x_values) + 3) - np.power(np.divide(np.array(x_values) + 1, np.array(x_values) + 2), 2), 0.5)#\frac{\epsilon_p +1}{\epsilon_p +3} - \frac{(\epsilon_p +1)^2}{(\epsilon_p +2)^2}
		elif distribution == 'exponential':
			predicted_mean = [sum([1/n for n in range(1, x+2)]) for x in x_values] #sum up to epsilon + 1
			predicted_std = np.power([sum([1/n**2 for n in range(1, int(x)+2)]) for x in x_values], 0.5)		
		lower_estimate = predicted_mean - predicted_std
		upper_estimate = predicted_mean + predicted_std
		ax[col].plot(x_values, predicted_mean, c='r', linestyle=(0, (3, 7)))
		ax[col].plot(x_values, lower_estimate, c='r', linestyle=(0, (3, 7)))
		ax[col].plot(x_values, upper_estimate, c='r', linestyle=(0, (3, 7)))
		ax[col].set_xlabel(r'NC evolvability $\epsilon_{NC}$')
		ax[col].set_xscale('log')
		ax[col].set_title(plotparam.type_structure_vs_label[type_structure] + ',\n' + distribution + ' PF-map')
		if distribution == 'uniform':
			ax[col].set_ylim(0, 1.03)
	custom_lines = [Line2D([0], [0], mfc=['k', 'r'][i], ls='', marker='o', label=['simulation', 'prediction'][i], mew=0, ms=5) for i in range(2)]
	ax[0].legend(handles=custom_lines, loc=4, frameon=False)
	#ax[1].axis('off')
[ax[i].annotate('ABCDEFGH'[i], xy=(0.0, 1.2), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(2)]
f.tight_layout()
f.savefig('./plots/peak_height_evolvability_summary' +'_iterations'+str(param.iterations_peaks)+ distribution+'.png', bbox_inches='tight', dpi=200)

####################################################################################################################################################
print('plot navigability vs # phenos and evolvability for all maps')
####################################################################################################################################################
for type_evolv in ['', 'mean', 'median']:
	for type_plot in ['', 'largest_component_']:
		f, ax = plt.subplots(figsize=(5.2, 2.9))
		cmap = matplotlib.colormaps['viridis']
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
		n_pheno_list, navigability_list, type_structure_list_data, pheno_evolvlist = [], [], [], []
		for type_structure in param.all_types_structue:

			NC_evolv_filename_oneNC_per_pheno = './data/'+type_plot+'NC_evolv_'+type_structure+ '_oneNC_per_pheno.csv'
			ph_evolv_filename = './data/'+type_plot+'ph_vs_evolv'+type_structure+ '.csv'
			nav_filename = './data/'+type_plot+'navigability'+type_structure+'_iterations'+str(param.iterations_nav)+'number_source_target_pairs'+str(param.number_source_target_pairs)+'.npy'
			if isfile(NC_evolv_filename_oneNC_per_pheno) and isfile(ph_evolv_filename) and isfile(nav_filename):
				single_NC_vs_evolv = load_dict(NC_evolv_filename_oneNC_per_pheno)
				if 'Fibonacci' in type_structure:
					if type_plot == '':
						print('navigability', type_structure, np.mean(np.load(nav_filename)))
					continue # this is separate plot
				navigability_list.append(np.mean(np.load(nav_filename)))
				type_structure_list_data.append(type_structure)
				ph_vs_evolv = load_dict(ph_evolv_filename)
				n_pheno_list.append(len([ph for ph in ph_vs_evolv.keys() if ph > 0]))
							
				assert len(single_NC_vs_evolv) == len(ph_vs_evolv)
				if type_evolv == '':
					pheno_evolvlist.append(gmean([e if e > 0 else 0.01 for e in ph_vs_evolv.values()]))
				elif type_evolv == 'mean':
					pheno_evolvlist.append(np.mean([e  for e in ph_vs_evolv.values()]))
				if type_evolv == 'median':
					pheno_evolvlist.append(np.median([e  for e in ph_vs_evolv.values()]))
				if type_plot == 'largest_component_':
					assert min([e for e in ph_vs_evolv.values()]) >= 1 # single NC vs evolv can be zero since one largest NC per phenotype retained, cutting some potential neighbours
			else:
				print('not found', type_plot, 'navigability files', type_structure)

		for j, type_structure in enumerate(type_structure_list_data):
			c = cmap(navigability_list[j])		
			mean_xpos = pheno_evolvlist[j]
			sc = ax.scatter([pheno_evolvlist[j]], [n_pheno_list[j]], c=c, s=10, marker='o')
			for cutoff in range(10):
				assert pheno_evolvlist[j] > ((n_pheno_list[j]*cutoff*0.1)**(1/n_pheno_list[j]) - 1) * n_pheno_list[j] or navigability_list[j] < cutoff*0.1
			if type_structure in ['RNA', 'DNA', 's_2_8', 'HP3x3x3s']:
				mean_xpos = mean_xpos * 1.21
			ax.annotate(plotparam.type_structure_vs_label[type_structure], (mean_xpos, n_pheno_list[j] * 1.17), fontsize=8, horizontalalignment='center')

		cb = f.colorbar(sm, ax=ax)
		cb.set_label('navigability')
		ax.set_xlabel(r'geometric mean of NC evolvability $\bar{\epsilon}_{NC}$')# + '\n'+r'(geometric mean, zeros treated as 0.01;'+'\n'+r'the 2 values represent 2 ways of resolving NC fragmentation)')
		ax.set_ylabel(r'number of phenotypes $n_p$')
		###
		n_data = np.power(10,np.linspace(np.log10(2), np.log10(max(n_pheno_list) * 10), 10**4))
		e_pred = [((n/10)**(1/n) - 1) * n for n in n_data]
		ax.plot(e_pred, n_data, c='k')
		###
		ax.set_ylim(7, max(n_pheno_list) * 2)
		ax.set_xlim(min(0.8, min(pheno_evolvlist)*0.5), max(pheno_evolvlist) * 2)
		ax.set_yscale('log')
		ax.set_xscale('log')
		f.tight_layout()
		f.savefig('./plots/'+type_plot+type_evolv+'navigability_scaling'+'_iterations'+str(param.iterations_nav)+'number_source_target_pairs'+str(param.number_source_target_pairs)+'.png', bbox_inches='tight', dpi=350)

####################################################################################################################################################
print('plot geometric vs arithmetic mean evolvability for all maps')
####################################################################################################################################################
f, ax = plt.subplots(figsize=(4, 3.5))
pheno_evolvlist1, pheno_evolvlist2, pheno_evolvlist3 = [], [], []
for type_structure in param.all_types_structue:
	if 'Fibonacci' in type_structure:
		continue # this is separate plot
	ph_evolv_filename = './data/'+'ph_vs_evolv'+type_structure+ '.csv'
	if isfile(ph_evolv_filename):
		assert 0 not in ph_vs_evolv
		ph_vs_evolv = load_dict(ph_evolv_filename)
		pheno_evolvlist1.append(gmean([e if e > 0 else 0.01 for e in ph_vs_evolv.values()]))		
		pheno_evolvlist2.append(np.mean([e for e in ph_vs_evolv.values()]))		
		pheno_evolvlist3.append(np.median([e for e in ph_vs_evolv.values()]))	
xlims = (0.5 * min(pheno_evolvlist1 + pheno_evolvlist2), 2 * max(pheno_evolvlist1 + pheno_evolvlist2))
###
ax.scatter(pheno_evolvlist1, pheno_evolvlist2, s=5, c='r')
ax.scatter(pheno_evolvlist1, pheno_evolvlist3, s=5, c='b')
print('corr different means', pearsonr(np.log10(pheno_evolvlist1), np.log10(pheno_evolvlist2)))
print('corr different means', pearsonr(np.log10(pheno_evolvlist1), np.log10(pheno_evolvlist3)))
ax.set_title('resolved NC fragmentation\nby connecting all NCs\nof a given phenotype')
ax.set_xlabel('geometric mean of NC evolvabilities\n(zero values treated as 0.01)')
ax.set_ylabel('arithmetic mean (red)/median (blue)\nof NC evolvabilities')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(xlims[0], xlims[1])
ax.set_xlim(xlims[0], xlims[1])
ax.plot(xlims, xlims, c='k', zorder=-5)
f.tight_layout()
f.savefig('./plots/evolv_mean'+'.png', bbox_inches='tight', dpi=350)"""
