function plot_number_of_iterations(alg_name, RES_TOL, SOL_TOL)

	% ===============
	% Set the paths
	% ===============

	HOME='~/src/robust_l0/GAGA_1_2_0/';

	NUMITER_PATH = fullfile(HOME, 'GAGA/RobustL0code/number_iterations/plots/');

	% ===============
	% Load the data
	% ===============

	load(['results_' alg_name '_S_smv.mat');

	% ===============
	% Get data
	% ===============

	for i = 1:length(results_cell)
		noise_level = results_cell{i}.noise_level;
		results = results_cell{i}.results;
		n_list = results_cell{i}.n_list;

		for j = 1:length(n_list)
			n = n_list(j);
			n_ind = results(:, 3) == n;

			ks = results(n_ind, 1);
			ms = results(n_ind, 2);
			ns = results(n_ind, 3);
			ds = results(n_ind, 5); 
			l1_error = results(n_ind, 7);
			total_time = results(n_ind, 10);
			iterations = results(n_ind, 13);

			deltas = ms./ns;
			rhos = ks./ms;

			% Get average iterations for each delta and each row

			unique_deltas = sort(unique(deltas));

			row = 1;
			data_list = zeros(length(deltas), 4);

			rho_step = 0.01;

			dg_n = length(unique_deltas);
			dg_m = 1/rho_step;
			data_grid = zeros(dg_m, dg_n);

			for ii = 1:length(unique_deltas)
				ind_delta = deltas == unique_deltas(ii);
				unique_rhos = sort(unique(rhos(ind_delta)));
				for jj = 1:length(unique_rhos)
					ind_rho = ind_delta & rhos == unique_rhos(jj);
					if any(ind_rho)
						mean_iter = mean(iterations(ind_rho));
						sd_iter = std(iterations(ind_rho));
						data_list(row, :) = [deltas(ii), rhos(jj), mean_iter, sd_iter];
						row = row + 1;
						data_grid(dg_m - jj, ii) = mean_iter;
					end
				end
			end

			row = row - 1;
			data_list = data_list(1:row, :);

			% Create figure

			figure;
			set(gcf, 'color', [1 1 1]);
			set(gca, 'Fontname', 'Times', 'Fontsize', 15);

			data_grid = log10(data_grid);

			%[X, Y] = meshgrid(1:size(data_grid, 2), 1:size(data_grid, 1));
			%[X2, Y2] = meshgrid(1:0.1:size(data_grid, 2), 1:0.1:size(data_grid, 1));
			[X, Y] = meshgrid(unique_deltas, rho_step:rho_step:1);
			[X2, Y2] = meshgrid(0:0.01:1, 0:0.01:1);

			outData = interp2(X, Y, data_grid, X2, Y2, 'linear');

			xx = X2(1, 1:end);
			yy = Y2(1:end, 1);

			imagesc(xx(:), yy(:), outData);

			colorbar;

			% Cosmetics

			set(gca,'YTickLabel',flipud(get(gca,'YTickLabel')));

			xlabel('\sigma'), ...
			ylabel('\rho*(\delta)');

			% Print title	

			nExp = log2(n);
			d = unique(ds);
			title({sprintf('%s: RES-TOL=%g, SOL-TOL=%1.3f', strrep(alg_name, '_', '-'), RES_TOL, SOL_TOL), ...
				sprintf('n=pow(2,%d), d=%d, noise=%1.3f', nExp, d, noise_level)});

			% Save figure

			noise_level_str = sprintf('%1.3f', noise_level);
			restol_str = sprintf('%1.3f', RES_TOL);
			soltol_str = sprintf('%1.3f', SOL_TOL);
			fig_name = ['iterations_', num2str(alg_name), '_n_', num2str(n), '_noise_', noise_level_str, '_RESTOL_', restol_str, '_SOLTOL_', soltol_str, '.pdf'];
			fig_path = fullfile(NUMITER_PATH, fig_name);
			print('-dpdf', fig_path);

		end

	end

end
