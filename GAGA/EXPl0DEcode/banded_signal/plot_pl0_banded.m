
clear all;

set(gca,'fontsize',14)

alg_list = {'parallel_l0'};
ns = [2^18];
nzs = [7];
ens_list = {'smv'};

destination = ['./plots/'];

% Phase transition of parallel_l0 for several values of d.

bands = 0:5:90;
bands = [0 5 10 15 20 25 30 40 50 60 70 90];
make_joint_transition_plots_by_band(alg_list{1}, 'smv', ns(1), nzs, bands, destination);

