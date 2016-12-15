
%%%% Set the paths

HOME='~/src/robust_l0/GAGA_1_2_0/';
GAGA_PATH=fullfile(HOME, 'GAGA/gaga/');
addpath(GAGA_PATH);
addpath(fullfile(HOME, 'GAGA/RobustL0code/phase_transitions/generate_data/'));
addpath(fullfile(HOME, 'GAGA/RobustL0code/phase_transitions/plot_data/'));

%%%%

alg_list = {'deterministic_robust_l0'};

GPUnumber = 0;
maxiter_str = '1k';
vecDistribution = 'gaussian';
band_percentage = 0.0;

n_list = [2^14];
nonZero_list = 7;

delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];
tests_per_k = 10;

noise_levels = [1e-3, 1e-2];
%RES_TOL = 2*1e-2;
RES_TOL = 1; % How many SD's from E[\|noise\|_2^2] is the residual allowed to differ
SOL_TOL = 1; % For deterministic_robust_l0 and robust_l0, this is how many SD's from E[\|noise\|_1] is the computed solution from the real one

fprintf('Generating data...\n');
generate_data_bottomup(alg_list, n_list, nonZero_list, delta_list, ...
	vecDistribution, RES_TOL, SOL_TOL, tests_per_k, GPUnumber, maxiter_str, ...
	band_percentage, noise_levels);

for i = 1:length(alg_list)
	alg_list{i} = strcat(alg_list{i}, '_S_smv');
end

fprintf('Processing data...\n');
process_smv_data(alg_list, vecDistribution);

fprintf('Adding logit... \n');
add_logit_data_smv(alg_list, SOL_TOL);

%%%%%%%%%%%%%%%%%%%%%%%
% PLOT
%%%%%%%%%%%%%%%%%%%%%%%

set(gca,'fontsize',14)

ns = n_list;
nzs = nonZero_list;
ens_list = {'smv'};

simplex_curve = 0;
addpath('../data/polytope/');

for i = 1:length(alg_list)
	dir = fullfile('../data/', alg_list{i});
	addpath(dir);
end

destination = fullfile('.');

% 50% phase transitions with all n on one plot.

make_transition_plots(alg_list, ens_list, destination);

% 50% phase transitions for given (ens, n, nonzeros)
% all algorithms shown on one plot.

if false
	for i = 1:length(ns)
		for j = 1:length(nzs)
			make_joint_transition_plots(alg_list, 'smv', ns(i), nzs(j), destination, simplex_curve);
			make_best_alg_plot(alg_list, 'smv', ns(i), nzs(j), destination);
			%make_best_alg_plot_0('smv', ns(i), nzs(j));
		end
	end
end
