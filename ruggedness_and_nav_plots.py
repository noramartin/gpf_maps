import numpy as np 
import networkx as nx 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from os.path import isfile
from scipy.stats import gmean, spearmanr
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import parameters as param
import parameters_plots as plotparam
from functions.navigability_functions import isundefinedpheno, load_dict


####################################################################################################################################################
####################################################################################################################################################
distribution = 'uniform'
####################################################################################################################################################

####################################################################################################################################################
print('ruggedness plot main text')
####################################################################################################################################################
####################################################################################################################################################
f, ax = plt.subplots(ncols  = 2, figsize=(5.2, 2.8), gridspec_kw={'width_ratios': [1, 0.51]})
xlims1 = [0.01, 0.01]
for type_structure in param.all_types_structue:
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
	NC_evolv_filename = './data/NC_evolv_'+type_structure+ '.csv'
	total_peak_size_filename = './data/total_peak_size_'+type_structure+'_iterations'+str(param.iterations_peaks)+distribution+ '.npy'
	NC_vs_size_filename = './data/NC_array'+ type_structure +'NC_vs_size.csv'
	if isfile(NC_evolv_filename) and isfile(total_peak_size_filename) and isfile(NC_vs_size_filename):
		NC_vs_evolv = load_dict(NC_evolv_filename)
		total_peak_size_list = np.load(total_peak_size_filename)
		NC_vs_size = load_dict(NC_vs_size_filename)
		predicted_size_peaks = np.sum([NC_vs_size[NC]/(e+1) for NC, e in NC_vs_evolv.items()])
		std, mean = np.std(total_peak_size_list), np.mean(total_peak_size_list)
		ax[0].errorbar([predicted_size_peaks/K**L,], [mean/K**L, ], yerr=std/K**L, c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')
		ax[0].set_xlabel(r'ruggedness prediction'+'\n'+r'from $|NC|$ '+'&'+ r' $\epsilon_{NC}$ data') 
		ax[0].set_ylabel('mean ruggedness\nin simulations')
		xlims1[0] = min(min(xlims1), mean/K**L)
		xlims1[1] = max(max(xlims1), mean/K**L + std/K**L)
	else:
		print('not found', NC_evolv_filename, total_peak_size_filename, NC_vs_size_filename)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
plot_lims = xlims1[0]* 0.5, xlims1[1]*3
ax[0].plot(plot_lims, plot_lims, c='k', lw=0.7)#, ls=':')
ax[0].set_xlim(plot_lims[0], plot_lims[1])
ax[0].set_ylim(plot_lims[0], plot_lims[1])
ax[-1].legend(handles=plotparam.legend_with_errorbar)
ax[-1].axis('off')
f.tight_layout()
f.savefig('./plots/ruggedness_main_fig'+'_iterations'+str(param.iterations_peaks) +distribution+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

####################################################################################################################################################
####################################################################################################################################################
print('ruggedness plot renormalised')
####################################################################################################################################################
####################################################################################################################################################
type_structure_vs_ruggedness, type_structure_vs_ruggedness_rescaledelet = {}, {}
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
				std, mean = np.std(total_peak_size_list), np.mean(total_peak_size_list)
				if not rescale_x_viable_genos:
					norm = K**L
				else:
					norm = sum([NC_vs_size[NC] for NC in NC_vs_evolv.keys() if NC > 0])
					assert norm <= K**L
				ax[1].errorbar([predicted_size_peaks/norm,], [mean/norm, ], yerr=std/norm, c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')
				ax[1].set_xlabel(r'prediction from $|NC|$ '+'&'+ r' $\epsilon_{NC}$:'+ '\nfraction of\ngenotypes that are peaks'+axis_label_addition) 
				ax[1].set_ylabel('simulation: fraction of\ngenotypes that are peaks'+axis_label_addition, fontsize='small')
				xlims1[0] = min(min(xlims1), mean/K**L)
				xlims1[1] = max(max(xlims1), mean/K**L + std/K**L)
				if not rescale_x_viable_genos and not exclude_zero_evolv:
					type_structure_vs_ruggedness[type_structure] = (mean/norm, std/norm)
				if rescale_x_viable_genos and not exclude_zero_evolv:
					type_structure_vs_ruggedness_rescaledelet[type_structure] = (mean/norm, std/norm)
				########
				ax[0].set_ylabel('simulation: fraction of\ngenotypes that are peaks'+axis_label_addition, fontsize='small')	
				if rescale_x_viable_genos and exclude_zero_evolv: 
					ax[0].errorbar([type_structure_vs_ruggedness_rescaledelet[type_structure][0],], [mean/norm, ], yerr=std/norm, xerr=type_structure_vs_ruggedness[type_structure][1],
					            c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')			
					ax[0].set_xlabel('simulation: fraction of\ngenotypes that are peaks\n(out of viable genotypes)') 					
				else:
					ax[0].errorbar([type_structure_vs_ruggedness[type_structure][0],], [mean/norm, ], yerr=std/norm, xerr=type_structure_vs_ruggedness[type_structure][1],
					            c=plotparam.type_structure_vs_color[type_structure], ms=7, marker='x', markeredgewidth=1.5, elinewidth=0.5, ecolor='grey')			
					ax[0].set_xlabel('simulation: fraction of\ngenotypes that are peaks\n(normalised by all genotypes)') 

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
		std, mean = np.std(total_peak_size_list), np.mean(total_peak_size_list)
		ax[0].errorbar([predicted_size_peaks/K**L,], [mean/K**L, ], yerr=std/K**L, c=plotparam.type_structure_vs_color[type_structure], ms=5, marker='o')
		ax[0].set_xlabel('analytic prediction:\nfraction of genotypes\nthat are peaks')
		ax[0].set_ylabel('simulation\nresult: fraction\nof genotypes\nthat are peaks')
		xlims1[0] = min(min(xlims1), min([mean/K**L, predicted_size_peaks/K**L]))
		xlims1[1] = max(max(xlims1), max([(mean +std)/K**L, predicted_size_peaks/K**L]))
ax[0].set_xscale('log')
ax[0].set_yscale('log')
plot_lims = xlims1[0]* 0.5, xlims1[1]*3
ax[0].plot(plot_lims, plot_lims, c='k', linestyle=':')
ax[0].set_xlim(plot_lims[0], plot_lims[1])
ax[0].set_ylim(plot_lims[0], plot_lims[1]) 
custom_lines = [Line2D([0], [0], mfc=plotparam.type_structure_vs_color[type_structure], linestyle='', marker='o', label=plotparam.type_structure_vs_label[type_structure], mew=0, ms=5) for type_structure in ['Fibonacci_12_3', 'Fibonacci_null_12_3']]
ax[1].legend(handles=custom_lines)
ax[1].axis('off')
f.tight_layout()
f.savefig('./plots/fibonacci_analytic_peaks_L'+str(L)+'_'+str(K)+'_iterations'+str(param.iterations_peaks) +'.png', bbox_inches='tight', dpi=250)

####################################################################################################################################################
print('plot peak height vs evolv')
####################################################################################################################################################
for plot_type in ['evolvability', 'size']:
	for distribution in ['exponential', 'uniform']:
		f, ax = plt.subplots(ncols  = 5, nrows=2, figsize=(15, 4.75))
		for i, type_structure in enumerate(param.all_types_structue):
			K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
			peak_evolv_height_filename = './data/peak_evolv_height_'+type_structure+'_iterations'+str(param.iterations_peaks)+ distribution+ '.csv'
			if isfile(peak_evolv_height_filename):
				df_peak_evolv_height = pd.read_csv(peak_evolv_height_filename, nrows=5*10**6)
				if len(df_peak_evolv_height['peak height']) > 10:
					sns.lineplot(data=df_peak_evolv_height, x='peak '+plot_type, y='peak height', color = 'k', estimator='mean', errorbar='sd', ax = ax[i//5, i%5], err_kws={'alpha': 0.4})
					if 'null' not in type_structure:
						print(type_structure, 'correlation peak height', plot_type, spearmanr(df_peak_evolv_height['peak height'].tolist(), df_peak_evolv_height['peak '+plot_type].tolist()))
					if plot_type == 'evolvability':
						x_values = [x for x in range(int(min([x for x in df_peak_evolv_height['peak ' +plot_type].tolist() if x > 0])), int(max(df_peak_evolv_height['peak ' +plot_type].tolist() ))+ 5)]
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
					if plot_type == 'size':
						ax[i//5, i%5].set_xlabel(r'NC size $|NC|$')
					if not (type_structure == 's_2_8' and plot_type == 'evolvability'):
			   			ax[i//5, i%5].set_xscale('log')
				ax[i//5, i%5].set_title(plotparam.type_structure_vs_label[type_structure])
				if distribution == 'uniform' and plot_type == 'evolvability':
					ax[i//5, i%5].set_ylim(0, 1.03)
			[ax[i//5, i%5].annotate('ABCDEFGHIJKLMN'[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(9)]
			ax[9//5, 9%5].axis('off')
		f.tight_layout()
		f.savefig('./plots/peak_height_'+plot_type+'_iterations'+str(param.iterations_peaks)+ distribution+'.png', bbox_inches='tight', dpi=200)


###################################################################################################################################################
print('plot peak height vs evolv - single plot ')
####################################################################################################################################################
f, ax = plt.subplots(figsize=(5, 2.5), ncols = 2)
plot_type = 'evolvability'
for col, (distribution, type_structure) in enumerate([('uniform', 'RNA_12'), ('exponential', 'RNA_12')]):
	K, L = param.type_structure_vs_K[type_structure], param.type_structure_vs_L[type_structure]
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
for type_plot in ['', 'largest_component_']:
	f, ax = plt.subplots(figsize=(5.2, 2.9))
	cmap = matplotlib.colormaps['viridis']
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
	n_pheno_list, navigability_list, type_structure_list_data, pheno_evolvlist, evolv_list_one_NC_per_pheno = [], [], [], [], []
	for type_structure in param.all_types_structue:
		if 'Fibonacci' in type_structure:
			continue # this is separate plot
		NC_evolv_filename_oneNC_per_pheno = './data/'+type_plot+'NC_evolv_'+type_structure+ '_oneNC_per_pheno.csv'
		ph_evolv_filename = './data/'+type_plot+'ph_vs_evolv'+type_structure+ '.csv'
		nav_filename = './data/'+type_plot+'navigability'+type_structure+'_iterations'+str(param.iterations_nav)+'number_source_target_pairs'+str(param.number_source_target_pairs)+'.npy'
		if isfile(NC_evolv_filename_oneNC_per_pheno) and isfile(ph_evolv_filename) and isfile(nav_filename):
			single_NC_vs_evolv = load_dict(NC_evolv_filename_oneNC_per_pheno)
			navigability_list.append(np.mean(np.load(nav_filename)))
			type_structure_list_data.append(type_structure)
			ph_vs_evolv = load_dict(ph_evolv_filename)
			n_pheno_list.append(len([ph for ph in ph_vs_evolv.keys() if ph > 0]))
			pheno_evolvlist.append(gmean([e if e > 0 else 0.01 for e in ph_vs_evolv.values()]))			
			assert len(single_NC_vs_evolv) == len(ph_vs_evolv)
			evolv_list_one_NC_per_pheno.append(gmean([e if e > 0 else 0.01 for e in single_NC_vs_evolv.values()]))
			if type_plot == 'largest_component_':
				assert min([e for e in ph_vs_evolv.values()]) >= 1 and min([e for e in single_NC_vs_evolv.values()]) >= 1
		else:
			print('not found', type_plot, 'navigability files', type_structure)

	for j, type_structure in enumerate(type_structure_list_data):
		c = cmap(navigability_list[j])
		ax.plot([pheno_evolvlist[j], evolv_list_one_NC_per_pheno[j]], [n_pheno_list[j],]*2, c=c, zorder=-1, linewidth=0.5)
		sc = ax.scatter([pheno_evolvlist[j]], [n_pheno_list[j]], c=c, marker='o', s=5)
		sc = ax.scatter([evolv_list_one_NC_per_pheno[j]], [n_pheno_list[j]], c=c, marker='o', s=5)
		mean_xpos = gmean([pheno_evolvlist[j], evolv_list_one_NC_per_pheno[j]])
		if type_structure in ['RNA', 'DNA', 's_2_8', 'HP3x3x3s']:
			mean_xpos = mean_xpos * 1.21
		ax.annotate(plotparam.type_structure_vs_label[type_structure], (mean_xpos, n_pheno_list[j] * 1.15), fontsize=8, horizontalalignment='center')

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
	f.savefig('./plots/'+type_plot+'navigability_scaling'+'_iterations'+str(param.iterations_nav)+'number_source_target_pairs'+str(param.number_source_target_pairs)+'.png', bbox_inches='tight', dpi=350)
####################################################################################################################################################
print('plot geometric vs arithmetic mean evolvability for all maps')
####################################################################################################################################################
f, ax = plt.subplots(ncols=2, figsize=(7.5, 3.5))
pheno_evolvlist1, pheno_evolvlist2, pheno_evolvlist3, evolv_list_one_NC_per_pheno1, evolv_list_one_NC_per_pheno2, evolv_list_one_NC_per_pheno3 = [], [], [], [], [], []
for type_structure in param.all_types_structue:
	if 'Fibonacci' in type_structure:
		continue # this is separate plot
	NC_evolv_filename_oneNC_per_pheno = './data/'+'NC_evolv_'+type_structure+ '_oneNC_per_pheno.csv'
	ph_evolv_filename = './data/'+'ph_vs_evolv'+type_structure+ '.csv'
	if isfile(NC_evolv_filename_oneNC_per_pheno) and isfile(ph_evolv_filename):
		single_NC_vs_evolv = load_dict(NC_evolv_filename_oneNC_per_pheno)
		ph_vs_evolv = load_dict(ph_evolv_filename)
		pheno_evolvlist1.append(gmean([e if e > 0 else 0.01 for e in ph_vs_evolv.values()]))		
		pheno_evolvlist2.append(np.mean([e for e in ph_vs_evolv.values()]))		
		pheno_evolvlist3.append(np.median([e for e in ph_vs_evolv.values()]))		
		assert len(single_NC_vs_evolv) == len(ph_vs_evolv)
		evolv_list_one_NC_per_pheno1.append(gmean([e if e > 0 else 0.01 for e in single_NC_vs_evolv.values()]))
		evolv_list_one_NC_per_pheno2.append(np.mean([e for e in single_NC_vs_evolv.values()]))
		evolv_list_one_NC_per_pheno3.append(np.median([e for e in single_NC_vs_evolv.values()]))	
xlims = (0.5 * min(pheno_evolvlist1 + pheno_evolvlist2+evolv_list_one_NC_per_pheno1+evolv_list_one_NC_per_pheno2), 2 * max(pheno_evolvlist1 + pheno_evolvlist2+evolv_list_one_NC_per_pheno1+evolv_list_one_NC_per_pheno2))
###
ax[0].scatter(pheno_evolvlist1, pheno_evolvlist2, s=5, c='r')
ax[0].scatter(pheno_evolvlist1, pheno_evolvlist3, s=5, c='b')
ax[0].set_title('resolved NC fragmentation\nby connecting all NCs\nof a given phenotype')
####
ax[1].scatter(evolv_list_one_NC_per_pheno1, evolv_list_one_NC_per_pheno2, s=5, c='r')
ax[1].scatter(evolv_list_one_NC_per_pheno1, evolv_list_one_NC_per_pheno3, s=5, c='b')
ax[1].set_title('resolved NC fragmentation\nby only keeping largest NC\nof each phenotype')
for i in range(2):
	ax[i].set_xlabel('geometric mean of NC evolvabilities\n(zero values treated as 0.01)')
	ax[i].set_ylabel('arithmetic mean (red)/median (blue)\nof NC evolvabilities')
	ax[i].set_xscale('log')
	ax[i].set_yscale('log')
	ax[i].set_ylim(xlims[0], xlims[1])
	ax[i].set_xlim(xlims[0], xlims[1])
	ax[i].plot(xlims, xlims, c='k', zorder=-5)
[ax[i].annotate('ABCDEFGH'[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(2)]

f.tight_layout()
f.savefig('./plots/evolv_mean'+'.png', bbox_inches='tight', dpi=350)

