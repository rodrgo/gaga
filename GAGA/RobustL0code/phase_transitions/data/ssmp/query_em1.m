

run('../init.m');

alg_list = {'ssmp'};

GPUnumber = 0;
maxiter_str = '3k';
vecDistribution = 'gaussian';
band_percentage = 0.0;

n_list = [2^14, 2^16, 2^18];
nonZero_list = 7;

delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];
tests_per_k = 15;

noise_levels = [1e-1];

RES_TOL = 1e-5;
SOL_TOL = 1;

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

