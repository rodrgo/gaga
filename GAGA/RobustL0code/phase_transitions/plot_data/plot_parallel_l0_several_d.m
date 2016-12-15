
clear all;

set(gca,'fontsize',14)

destination = ['../plots/ccs/'];

% Phase transition of parallel_l0 for several values of d.

alg_list = {'parallel_l0'};
ns = [2^18];
nzs = [5 7 9 11 13 15 17 19];

make_joint_transition_plots_by_d(alg_list{1}, 'smv', ns(1), nzs, destination);


