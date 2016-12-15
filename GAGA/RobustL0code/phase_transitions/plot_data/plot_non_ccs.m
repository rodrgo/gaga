
clear all;

set(gca,'fontsize',14)

alg_list = {'ALPS', 'CGIHT', 'CGIHTprojected', 'CGIHTrestarted', 'CSMPSP', 'FIHT', 'HTP', 'NIHT'};

ns = [2^18];
nzs = [7];
ens_list = {'smv'};
simplex_curve = 0;

addpath('../data/non_ccs/')
destination = ['../plots/non_ccs/'];

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

