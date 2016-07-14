

% Run in polaris
run('../init.m');

alg_list = cell(1,1);
alg_list{1} = 'parallel_l0';
GPUnumber = 0;
maxiter_str = '1k';

n_list = [2^18];
nonZero_list = [7];
delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];

Tol = 1e-5;
vecDistribution = 'gaussian';
band_percentage = 0.15;
tests_per_k = 10;

fprintf('Generating data...\n');
generate_data_bottomup(alg_list, n_list, nonZero_list, delta_list, ...
	vecDistribution, Tol, tests_per_k, GPUnumber, maxiter_str, band_percentage);

for i = 1:length(alg_list)
	alg_list{i} = strcat(alg_list{i}, '_S_smv');
end

fprintf('Processing data...\n');
process_smv_data(alg_list);

fprintf('Adding logit... \n');
add_logit_data_smv(alg_list);
