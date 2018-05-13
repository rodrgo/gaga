
algorithms = {'robust_l0', 'robust_l0_adaptive', 'robust_l0_trans', 'robust_l0_adaptive_trans', 'ssmp_robust', 'smp_robust', 'cgiht_robust'};

PLOT_TOL_CURVE = true;

load_clips;
SOL_TOL = 1;

% Font size
fs = [];
fs.title = 20;
fs.legend = 17;
fs.axis = 20;
fs.ticks = 20;

MS = 'MarkerSize';
LW = 'LineWidth';

s_list = '+o*xsdh^v><p.';

% =============
% Extract data
% =============

algs = [];
ks = [];
ms = [];
ns = [];
ds = [];
errs1 = []; 
errs2 = []; 
times = [];
iters = [];
noise_levels = [];

fnames = {};

for i = 1:length(algorithms) 

	% Query files in algorithms{i} directory

	files = strsplit(ls(strcat(algorithms{i}, '/gpu_data_smv*')));

	for j = 1:length(files)
		if ~isempty(files{j})

			fprintf('Reading %s\n', files{j});
			
			fidi = fopen(files{j});
			data = textscan(fidi, '%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');

			algs = [algs; strrep(strrep(data{1}, '_S_smv', ''), '_', '-')];
			ks = [ks; data{4}];
			ms = [ms; data{7}];
			ns = [ns; data{10}];
			ds = [ds; data{15}];
			errs1 = [errs1; data{19}];
			errs2 = [errs2; data{21}];
			times = [times; data{25}/1000]; % want time in seconds
			iters = [iters; data{33}];
			noise_levels = [noise_levels; data{47}];

		end
	end

end

% ----------------------
% Filter out for sigma < 0.8
% ----------------------

filter_ind = noise_levels <= 0.1;

algs	= algs(filter_ind);
ks 	= ks(filter_ind);
ms	= ms(filter_ind);
ns	= ns(filter_ind);
ds	= ds(filter_ind);
errs1	= errs1(filter_ind);
errs2	= errs2(filter_ind);
imes	= times(filter_ind);
iters	= iters(filter_ind);
noise_levels	= noise_levels(filter_ind);

% ----------------------
% Create dictionary of algorithm names
% ----------------------

algNames = unique(algs);

algCodes = zeros(size(algs));
for i = 1:length(algNames)
	algCodes(ismember(algs, algNames{i})) = i;
end

fprintf('%d algorithms in data\n', length(algNames));

% ----------------------
% Create convergence indicator vector
% ----------------------

% Initialise converged to 'false'
converged = ones(size(errs2)) == 0;

is_robust_l0 = ismember(algs, {'robust-l0', 'robust-l0-adaptive', 'robust-l0-trans', 'robust-l0-adaptive-trans', 'ssmp-robust', 'smp-robust', 'cgiht-robust'});

mean_err1 = ms.*noise_levels*sqrt(2/pi);
sd_err1 = sqrt(ms).*noise_levels*sqrt(1 - 2/pi);
mean_signal_norm = ks.*sqrt(2/pi);

robust_l0_TOL = min((mean_err1 + SOL_TOL*sd_err1)./mean_signal_norm, UPPER_BOUND_CLIP);

converged(is_robust_l0) = errs1 <= robust_l0_TOL;

% ----------------------
% Store all data
% ----------------------

data = [];
data.algCodes = algCodes;
data.converged = converged;
data.ns = ns;
data.deltas = ms./ns;
data.rhos = ks./ms;
data.iters = iters;
data.times = times;
data.noise_levels = noise_levels;

%data = [algCodes, converged, ks, ms, ns, deltas, rhos, iters, times, noise_levels];


% ======================
% Get transition probabilities
% ======================

unique_algCodes = unique(data.algCodes);
unique_deltas = unique(data.deltas); 
unique_ns = unique(data.ns);

assert(length(unique_ns) == 1, 'Code is written for only one n');
assert(length(unique(ds)) == 1, 'Code is written for only one d');

d = ds(1);
n = unique_ns(1);

% tuples are [alg_code, delta, noise_level, rho, prob_conv, avg_time]
pt_data = {};

for i = 1:length(unique_algCodes)
	algCode = unique_algCodes(i);
	ind_1 = data.algCodes == algCode;
	alg_name = algNames{i}; 

	for j = 1:length(unique_deltas)
		delta = unique_deltas(j);
		ind_2 = ind_1 & (data.deltas == delta);

		% We want a phase transition curve for each algorithm and each delta

		unique_noiseLevels = unique(data.noise_levels(ind_2));

		% curve_at_sigma = {rho, prob_convergence, average_time}

		for l = 1:length(unique_noiseLevels)
			noise_level = unique_noiseLevels(l);

			ind_at_sigma = ind_2 & (data.noise_levels == noise_level);
			rhos_at_sigma = sort(unique(data.rhos(ind_at_sigma)));
			
			% rhos for this tuple of (algCode, delta, noise_level)

			prob_convergence = zeros(size(rhos_at_sigma));
			average_time = zeros(size(rhos_at_sigma));

			for r = 1:length(rhos_at_sigma)
				rho = rhos_at_sigma(r);
				ind_at_sigma_rho = ind_at_sigma & (data.rhos == rho);
				prob_convergence(r) = sum(data.converged(ind_at_sigma_rho))/nnz(ind_at_sigma_rho);
				average_time(r) = sum(data.times(ind_at_sigma_rho))/nnz(ind_at_sigma_rho);
			end

			% Summary of prob_convergence and average_time

			% Get largest rho such that probability of success is one.

			r_max_ind = max(find(prob_convergence > 0.5, 1, 'last'));
			if isempty(r_max_ind)
				fprintf(sprintf('Warning: %s did not converge at some parameters\n', change_names(alg_name)));
				r_max_ind = 1;
			end
			rho_max = rhos_at_sigma(r_max_ind);
			rho_max_time = average_time(r_max_ind);

			pt_data{end + 1} = [algCode, delta, noise_level, rho_max, rho_max_time];

		end

	end

end

pt_matrix = zeros(length(pt_data), 5);

for i = 1:length(pt_data)
	pt_matrix(i, :) = pt_data{i};
end

% ======================
% Plot phase transition
% ======================

MS = 'MarkerSize';
max_MS = 15;

for i = 1:length(unique_algCodes)
	algCode = unique_algCodes(i);
	alg_name = algNames{i}; 

	% Create figure for probabilities

	hold off;

	figure;
	hold on;
	set(gcf, 'color', [1 1 1]);
	set(gca, 'Fontname', 'Times', 'Fontsize', 15);

	handles = [];
	labels = {};

	for j = 1:length(unique_deltas)
		delta = unique_deltas(j);

		% For this value of algCode and this delta, extract a summary of
		% time or prob_convergence

		ind = pt_matrix(:, 1) == algCode & pt_matrix(:, 2) == delta;

		sigmas = pt_matrix(ind, 3);
		rhos = pt_matrix(ind, 4);

		linestyle = strcat('-', s_list(i));
		handles(end + 1) = semilogx(sigmas, rhos, linestyle, LW, 2, MS, 10);
		hold on;
		labels{end + 1} = ['\delta = ', sprintf('%1.2f', delta)];

	end

	legend(handles, labels, 'FontSize', fs.legend, 'Location', 'SouthWest');
	xlabel('\sigma', 'FontSize', fs.axis), ...
	ylabel('\rho*(\delta)', 'FontSize', fs.axis);

	% Print title	

	nExp = log2(n);
	title(sprintf('%s with n = 2^{%s} and d = %d', change_names(alg_name), num2str(nExp), d), 'FontSize', fs.title);

	hold off;

	% Save figure

	fig_name = ['plots/rho_vs_noise_probs_', change_names(alg_name), '.pdf'];

	print('-dpdf', fig_name);

end

% ======================
% Plot timing
% ======================

MS = 'MarkerSize';
max_MS = 15;

for i = 1:length(unique_algCodes)
	algCode = unique_algCodes(i);
	alg_name = algNames{i}; 

	% Create figure for probabilities

	hold off;

	figure;
	set(gcf, 'color', [1 1 1]);
	set(gca, 'Fontname', 'Times', 'Fontsize', 15);
	

	handles = [];
	labels = {};

	for j = 1:length(unique_deltas)
		delta = unique_deltas(j);

		% For this value of algCode and this delta, extract a summary of
		% time or prob_convergence

		ind = pt_matrix(:, 1) == algCode & pt_matrix(:, 2) == delta;

		sigmas = pt_matrix(ind, 3);
		rho_time = pt_matrix(ind, 5);

		handles(end + 1) = loglog(sigmas, rho_time, '.-');
		hold on;
		labels{end + 1} = ['\delta = ', sprintf('%1.2f', delta)];

	end


	legend(handles, labels, 'FontSize', fs.legend);
	xlabel('\sigma', 'FontSize', fs.axis), ...
	ylabel('Average time (sec) at \rho*(\delta)', 'FontSize', fs.axis);

	% Print title	

	nExp = log2(n);
	title(sprintf('%s with n = 2^{%d} and d = %d', change_names(alg_name), nExp, d), 'FontSize', fs.title);

	hold off;

	% Save figure

	fig_name = ['plots/rho_vs_noise_time_', change_names(alg_name), '.pdf'];

	print('-dpdf', fig_name);

end


% ======================
% Plot phase transition facet by delta
% ======================

MS = 'MarkerSize';
max_MS = 15;
for j = 1:length(unique_deltas)
	delta = unique_deltas(j);

	% Create figure for probabilities

	hold off;

	figure;
	set(gcf, 'color', [1 1 1]);
	set(gca, 'Fontname', 'Times', 'Fontsize', 15);

	handles = [];
	labels = {};

	for i = 1:length(unique_algCodes)
		algCode = unique_algCodes(i);
		alg_name = algNames{i}; 

		% For this value of algCode and this delta, extract a summary of
		% time or prob_convergence

		ind = pt_matrix(:, 1) == algCode & pt_matrix(:, 2) == delta;

		sigmas = pt_matrix(ind, 3);
		rhos = pt_matrix(ind, 4);

		linestyle = strcat('-', s_list(i));
		handles(end + 1) = semilogx(sigmas, rhos, linestyle, LW, 2, MS, 10);
		hold on;
		labels{end + 1} = change_names(alg_name);

	end

	if PLOT_TOL_CURVE
		m = delta*n;
		slope = (1 + SOL_TOL/sqrt(m)*sqrt((pi-2)/2));
		handles(end + 1) = semilogx(sigmas, slope*sigmas, 'k--');
		hold on;
		labels{end + 1} = ['TOL thresh'];
	end

	if floor(delta*100)/100 == 0.2
		legend(handles, labels, 'FontSize', fs.legend, 'Location', 'SouthWest');
	end
	xlabel('\sigma', 'FontSize', fs.axis), ...
	ylabel('\rho*(\delta)', 'FontSize', fs.axis);

	% Tick size
	xt = get(gca, 'XTick');
	set(gca, 'FontSize', fs.ticks);

	xt = get(gca, 'YTick');
	set(gca, 'FontSize', fs.ticks);

	% Print title	

	nExp = log2(n);
	title(['\delta = ' sprintf('%1.2f with n = 2^{%s} and d = %d', delta, num2str(floor(nExp)), d)], 'FontSize', fs.title);

	hold off;

	% Save figure

	fig_name = ['plots/rho_vs_noise_probs_delta_' sprintf('%1.2f', delta) '.pdf'];

	print('-dpdf', fig_name);

end

% ======================
% Plot timing facet by delta
% ======================

MS = 'MarkerSize';
max_MS = 15;

for j = 1:length(unique_deltas)
	delta = unique_deltas(j);

	% Create figure for probabilities

	hold off;

	figure;
	set(gcf, 'color', [1 1 1]);
	set(gca, 'FontName', 'Times', 'FontSize', 15);
	

	handles = [];
	labels = {};

	for i = 1:length(unique_algCodes)
		algCode = unique_algCodes(i);
		alg_name = algNames{i}; 

		% For this value of algCode and this delta, extract a summary of
		% time or prob_convergence

		ind = pt_matrix(:, 1) == algCode & pt_matrix(:, 2) == delta;

		sigmas = pt_matrix(ind, 3);
		rho_time = pt_matrix(ind, 5);

		handles(end + 1) = loglog(sigmas, rho_time, strcat('-', s_list(i)), LW, 2, MS, 10);
		hold on;
		labels{end + 1} = change_names(alg_name);

	end

	legend(handles, labels, 'FontSize', fs.legend);
	xlabel('\sigma', 'FontSize', fs.axis), ...
	ylabel('Average time (sec) at \rho*(\delta)', 'FontSize', fs.axis);

	% Print title	

	nExp = log2(n);
	title(['\delta=' sprintf('%1.2f with n = 2^{%d} and d = %d', delta, nExp, d)], 'FontSize', fs.title);

	hold off;

	% Save figure

	fig_name = ['plots/rho_vs_noise_time_delta_' sprintf('%1.2f', delta)];

	print('-dpdf', [fig_name '.pdf']);
	print('-depsc', [fig_name '.eps']);

end
