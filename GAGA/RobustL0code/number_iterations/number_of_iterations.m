function number_of_iterations(alg_name, noise_levels, RES_TOL, SOL_TOL, GPUnumber, maxiter_str, l0_thresh)

	%GPUnumber = 0;
	%noise_levels = [1e-2];
	%
	%RES_TOL = [1e-1]; % How many SD's from E[\|noise\|_2^2] is the residual allowed to differ
	%SOL_TOL = [1e-1]; % For deterministic_robust_l0 and robust_l0, this is how many SD's from E[\|noise\|_1] is the computed solution from the real one

	if nargin < 6
		l0_thresh = 2;
	end

	if nargin < 5;
		maxiter_str = '1k';
	end

	% ===============
	% Set the paths
	% ===============

	HOME='~/src/robust_l0/GAGA_1_2_0/';
	GAGA_PATH=fullfile(HOME, 'GAGA/gaga/');

	addpath(GAGA_PATH);
	addpath(fullfile(HOME, 'GAGA/RobustL0code/phase_transitions/generate_data'));

	% ===============
	% Generate the data
	% ===============

	alg_list = {alg_name};

	vecDistribution = 'gaussian';
	band_percentage = 0.0;

	delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];
	tests_per_k = 10;

	n_list = [2^14];
	nonZero_list = 7;

	fprintf('Generating data...\n');
	generate_data_bottomup(alg_list, n_list, nonZero_list, delta_list, ...
		vecDistribution, RES_TOL, SOL_TOL, tests_per_k, GPUnumber, maxiter_str, ...
		band_percentage, noise_levels, l0_thresh);

	for i = 1:length(alg_list)
		alg_list{i} = strcat(alg_list{i}, '_S_smv');
	end

	fprintf('Processing data...\n');
	process_smv_data(alg_list, vecDistribution);

	fprintf('Adding logit... \n');
	add_logit_data_smv(alg_list, SOL_TOL);

end
