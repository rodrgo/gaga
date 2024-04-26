run('../init.m');

alg_list = {'robust_l0_adaptive_trans'};

GPUnumber = 4;
maxiter_str = '1k';
vecDistribution = 'gaussian';
band_percentage = 0.0;

n_list = [2^14, 2^16, 2^18];
nonZero_list = 7;

delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];
tests_per_k = 10;

noise_levels = [1e-3, 1e-2, 1e-1];
RES_TOL = 0;
SOL_TOL = 1;


for i = 1:length(alg_list)
	alg_list{i} = strcat(alg_list{i}, '_S_smv');
end

fprintf('Processing data...\n');
process_smv_data(alg_list, vecDistribution);

fprintf('Adding logit... \n');
add_logit_data_smv(alg_list, SOL_TOL);
