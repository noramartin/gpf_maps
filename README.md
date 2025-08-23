# GPF maps
This is the code for the preprint "Predicting the topography of fitness landscapes from the structure of genotype-phenotype maps" (Malvika Srivastava, Ard A. Louis, Nora S. Martin)

The code uses Python 3 and standard Python packages (matplotlib, networkx, pandas, seaborn, scipy etc.).

The code relies on the following datasets:
- The computational biophysical GP maps from Greenbury et al. (Nat Eco Evo 2022) - the "gp_maps" folder needs to be downloaded from https://github.com/sgreenbury/gp-maps-nav
- The GP map based on RBP data curated by Payne and Wagner (PNAS 2018) - this data needs to be read in first, see the "RNA_map_construction" folder.

The code uses the following main references:
- The Fibonacci model and its low-evolvability version are implemented in the code (see paper for details). This model is adapted from Weiß & Ahnert (J R Soc Interface 2018), who in turn built on Greenbury and Ahnert's binary Fibonacci model (J R Soc Interface 2015).
- The method for identifying NCs follows Grüner, W., Giegerich, R., Strothmann, D., Reidys, C., Weber, J., Hofacker, I.L., Stadler, P.F. and Schuster, P., 1996. Analysis of RNA sequence structure maps by exhaustive enumeration II. Structures of neutral networks and shape space covering. Monatshefte für Chemie/Chemical Monthly, 127(4), pp.375-389.
- Some functions for GP map analyses and plots are adapted from the authors' previous papers.

The scripts in the main folder contain the following content:
- analyse_greenbury_GPmaps.py reads in all GP maps used in the main text, extracts NCs, builds the NC graph and generates data on peaks and navigabilities. It also generates ruggedness data for two different perturbations on the maps (see SM): one with reduced dimensionality and one shuffled GP map. This script needs to be run for each GP map, with the map label as an argument (labels are listed in the "parameters.py" file).
- fibonacci_peaks.py - this script generates additional data for plots that are specific to the Fibonacci model or synthetic NC graphs. It has three parameters: the sequence length L, alphabet size K and a parameter "type_analysis" controlling which parts of the script to run. For parameter (type_analysis=-1), it runs tests. For parameter (type_analysis=3), the script counts the number of paths of length k for SM section "Scaling of minimum evolvability required for navigability". This analysis should be run for shorter sequence lengths and alphabet sizes since it is computationally expensive (here L=7, K=3). For parameter (type_analysis = 4), the script computes navigabilities for a family of Fibonacci GP maps, with increasing numbers of phenotypes replaced by the undefined phenotype. This data is used in the SM section "Navigable vs. non-navigable GP maps in a sample of GP maps based on the Fibonacci model". We ran this part of the script for $5\leq L \leq 13$ and $2\leq K\leq 5$, except maps with L/K combinations of 12/4, 13/4, 10/5, 11/5, 12/5, 13/5, which were computationally unfeasible. For parameter (type_analysis = 5), the script generates synthetic NC graphs (here L and K do not matter) and evaluates their navigabilities, thus producing the data for SM section "Navigable vs. non-navigable synthetic NC graphs".
- fibonacci_plots.py - this script takes the data produced by the fibonacci_peaks.py script, and produces the corresponding plots.
- NC_graph_plots.py - this script reads in files from "analyse_greenbury_GPmaps.py", prints the information for SM Table 1 (including checks against Greenbury et al.'s numbers, where they report the same quantities for their GP maps), and generates the SM plots characterising NC graphs (NC sizes, evolvabilities etc.).
- ruggedness_and_nav_plots.py - this script reads in data from "analyse_greenbury_GPmaps.py", and creates all plots focussing on peaks and navigability, except those produced by the fibonacci_plots.py script.
- parameters_plots.py and parameters.py are just parameter files.
