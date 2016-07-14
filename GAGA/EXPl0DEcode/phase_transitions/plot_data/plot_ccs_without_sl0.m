
clear all;

set(gca,'fontsize',14)

alg_list = {'smp', 'ssmp', 'er', 'parallel_lddsr', 'parallel_l0'};
ns = [2^18];
nzs = [7];
ens_list = {'smv'};
simplex_curve = 1;

addpath('../data/polytope/');
for i = 1:length(alg_list)
	dir = strcat('../data/', alg_list{i});
	addpath(dir);
end
destination = ['../plots/ccs_no_sl0/'];

% 50% phase transitions for given (ens, n, nonzeros)
% all algorithms shown on one plot.

for i = 1:length(ns)
	for j = 1:length(nzs)
		make_joint_transition_plots(alg_list, 'smv', ns(i), nzs(j), destination, simplex_curve);
		make_best_alg_plot(alg_list, 'smv', ns(i), nzs(j), destination);
		%make_best_alg_plot_0('smv', ns(i), nzs(j));
	end
end



