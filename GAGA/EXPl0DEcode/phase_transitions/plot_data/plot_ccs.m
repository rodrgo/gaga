
clear all;

set(gca,'fontsize',14)

alg_list = {'smp', 'ssmp', 'er', 'parallel_lddsr', 'parallel_l0', 'serial_l0'};
ns = [2^14 2^16 2^18];
nzs = [7];
ens_list = {'smv'};
simplex_curve = 1;

addpath('../data/polytope/');
for i = 1:length(alg_list)
	dir = strcat('../data/', alg_list{i});
	addpath(dir);
end
destination = ['../plots/ccs/'];

% 50% phase transitions with all n on one plot.

make_transition_plots(alg_list, ens_list, destination);

% 50% phase transitions for given (ens, n, nonzeros)
% all algorithms shown on one plot.

for i = 1:length(ns)
	for j = 1:length(nzs)
		make_joint_transition_plots(alg_list, 'smv', ns(i), nzs(j), destination, simplex_curve);
		make_best_alg_plot(alg_list, 'smv', ns(i), nzs(j), destination);
		%make_best_alg_plot_0('smv', ns(i), nzs(j));
	end
end

% Phase transition of parallel_l0 for several values of d.

alg_list = {'parallel_l0'};
ns = [2^18];
nzs = [5 7 9 11 13 15 17 19];

make_joint_transition_plots_by_d(alg_list{1}, 'smv', ns(1), nzs, destination);


