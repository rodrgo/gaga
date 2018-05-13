run('../init.m');

alg_list = {'ssmp_robust'};

GPUnumber = 0;
maxiter_str = '4k';
vecDistribution = 'gaussian';
band_percentage = 0.0;

n_list = [2^14 2^16 2^18];
nonZero_list = 7;

delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];
tests_per_k = 11;

noise_levels = [1e-3, 1e-2, 1e-1];
RES_TOL = 0; % How many SD's from E[\|noise\|_2^2] is the residual allowed to differ
SOL_TOL = 1; % For deterministic_robust_l0 and robust_l0, this is how many SD's from E[\|noise\|_1] is the computed solution from the real one

%fprintf('Generating data...\n');
%generate_data_bottomup(alg_list, n_list, nonZero_list, delta_list, ...
%	vecDistribution, RES_TOL, SOL_TOL, tests_per_k, GPUnumber, maxiter_str, ...
%	band_percentage, noise_levels);

for i = 1:length(alg_list)
	alg_list{i} = strcat(alg_list{i}, '_S_smv');
end

fprintf('Processing data...\n');
process_smv_data(alg_list, vecDistribution);

fprintf('Adding logit... \n');
add_logit_data_smv(alg_list, SOL_TOL);

