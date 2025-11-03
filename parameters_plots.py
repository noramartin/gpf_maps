import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


type_structure_vs_K = {'HP_20': 2, 'HP3x3x3s': 2, 'HP5x5s': 2, 'HP25': 2, 'RNA_12': 4, 's_2_8': 8, 'RNA': 4, 'Fibonacci_12_3': 3, 'Fibonacci_null_12_3': 3} #, 'DNA': 4
all_types_structue = [type_structure for type_structure in type_structure_vs_K]

####################################################################################################################################################
## Greenbury data on maps for checks (from Greenbury theisis and Nat Ecol Evo 2022)
####################################################################################################################################################
type_structure_vs_no_NCs = {'HP25': 148254, 'HP5x5s': 6785, 'HP3x3x3s': 732157, 's_2_8': 1347, 'RNA_12': 645,  }#'RNA_15':12526
type_structure_vs_navigability = {'HP_20': 0.004, 'HP3x3x3s': 0.669, 'HP5x5s': 0.995, 'HP25': 0.013, 'RNA_12': 0.966, 's_2_8': 0.913} #####based on extended data table at end of article
type_structure_vs_fraction_unbound = {'HP_20': 0.976, 'HP3x3x3s': 0.939, 'HP5x5s': 0.816, 'HP25': 0.977, 'RNA_12': 0.854, 's_2_8': 0.537}
type_structure_vs_nph = {'HP_20': 5311, 'HP3x3x3s': 49808, 'HP5x5s': 550, 'HP25': 107337, 'RNA_12': 58, 's_2_8': 14}
####################################################################################################################################################
## info for plots
####################################################################################################################################################
type_structure_vs_label = {type_structure: type_structure for type_structure in all_types_structue}
type_structure_vs_label['Fibonacci_12_3'] = 'Fibonacci'
type_structure_vs_label['Fibonacci_null_12_3'] = 'LE Fibonacci'
type_structure_vs_label['RNA'] = 'RNA-binding'
type_structure_vs_label['RNA_12'] = 'RNA structure'
type_structure_vs_label['s_2_8'] = 'self-assembly'
type_structure_vs_color =  {all_types_structue[N]: color for N, color in enumerate(sns.color_palette(None, len(all_types_structue)))}
legend_without_Fibonaccinull = [Line2D([0], [0], mfc=type_structure_vs_color[type_structure], ls='', marker='o', label=type_structure_vs_label[type_structure], mew=0, ms=5) for type_structure in type_structure_vs_K if 'null' not in type_structure]
legend = [Line2D([0], [0], mfc=type_structure_vs_color[type_structure], ls='', marker='o', label=type_structure_vs_label[type_structure], mew=0, ms=5) for type_structure in type_structure_vs_K]
legend_with_errorbar = [x for x in legend] + [Line2D([0], [0], c='grey', lw=0.5, ls='-', marker=None, label=r'$20^{th}$-$80^{th}$ percent.', mew=0, ms=5)]





