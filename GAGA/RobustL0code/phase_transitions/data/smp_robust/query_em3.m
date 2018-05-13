

run('../init.m');

alg_list = {'smp_robust'};

GPUnumber = 3;
maxiter_str = '1k';
vecDistribution = 'gaussian';
band_percentage = 0.0;

n_list = [2^14, 2^16, 2^18];
nonZero_list = 7;

delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];
tests_per_k = 11;

%noise_levels = [1e-3, 1e-2, 1e-1];
noise_levels = [1e-3];

RES_TOL = 0;
SOL_TOL = 1;

fprintf('Generating data...\n');
generate_data_bottomup(alg_list, n_list, nonZero_list, delta_list, ...
	vecDistribution, RES_TOL, SOL_TOL, tests_per_k, GPUnumber, maxiter_str, ...
	band_percentage, noise_levels);

