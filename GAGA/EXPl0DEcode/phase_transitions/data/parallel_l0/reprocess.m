
run('../init.m');

alg_list = {'parallel_l0'};

for i = 1:length(alg_list)
	alg_list{i} = strcat(alg_list{i}, '_S_smv');
end

fprintf('Processing data...\n');
process_smv_data(alg_list);

fprintf('Adding logit... \n');
add_logit_data_smv(alg_list);

