
clear all;

set(gca,'fontsize',14)

ns = [2^18];
nzs = [7];
ens_list = {'smv'};

simplex_curve = 0;
addpath('../data/polytope/');

alg_list_ccs = {'parallel_l0', 'er', 'parallel_lddsr', 'smp', 'ssmp'};
for i = 1:length(alg_list_ccs)
	dir = strcat('../data/', alg_list_ccs{i});
	addpath(dir);
end

alg_list_geo = {'ALPS', 'CGIHT', 'CGIHTprojected', 'CGIHTrestarted', 'CSMPSP', 'FIHT', 'HTP', 'NIHT'}; 
addpath('../data/non_ccs/')

alg_list = {};
for i = 1:length(alg_list_ccs)
	alg_list{end + 1} = alg_list_ccs{i};
end
for i = 1:length(alg_list_geo)
	alg_list{end + 1} = alg_list_geo{i};
end

destination = ['../plots/all/'];

% 50% phase transitions with all n on one plot.

%make_transition_plots(alg_list, ens_list, destination);

% 50% phase transitions for given (ens, n, nonzeros)
% all algorithms shown on one plot.

for i = 1:length(ns)
	for j = 1:length(nzs)
		make_joint_transition_plots(alg_list, 'smv', ns(i), nzs(j), destination, simplex_curve);
		make_best_alg_plot(alg_list, 'smv', ns(i), nzs(j), destination);
		%make_best_alg_plot_0('smv', ns(i), nzs(j));
	end
end

