
clear all;

algorithms = {'robust_l0', 'robust_l0_adaptive', 'robust_l0_trans', 'robust_l0_adaptive_trans'};
%, 'deterministic_robust_l0'};
statistic = 'mean'; % mean, median, mean_converged

load_clips;

% Font size
fs = [];
fs.title = 20;
fs.legend = 17;
fs.axis = 20;
fs.ticks = 20;

s_list = '+o*xsdh^v><p.';

% ===============
% Set tolerances
% ==============

SOL_TOL = 1;
RES_TOL = 0;

% ===============
% Extract data
% ==============

algs = [];
ks = [];
ms = [];
ns = [];
ps = [];
errs = [];
errs_1 = [];
times = [];
iters = [];
noise_levels = [];

for i = 1:length(algorithms) 
	
	filePaths = strsplit(ls(strcat(algorithms{i}, '/gpu_data_smv*')));

	for j = 1:(length(filePaths)-1)  % carriage return gets split

		filePaths{j}
		fidi = fopen(filePaths{j});
		data = textscan(fidi,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');

		algs = [algs; strrep(strrep(data{1}, '_S_smv', ''), '_', '-')];
		ks = [ks; data{4}];
		ms = [ms; data{7}];
		ns = [ns; data{10}];
		ps = [ps; data{15}];
		errs_1 = [errs; data{19}];
		errs = [errs; data{21}];
		times = [times; data{25}/1000]; % want time in seconds
		iters = [iters; data{33}];
		noise_levels = [noise_levels; data{47}];
		fclose(fidi);

	end

end


% ===============
% Extract data
% ==============

algNames = unique(algs);
nAlgs = length(algNames);
algCodeDict = 1:nAlgs;
algCodes = zeros(size(algs));
for i = 1:length(algs)
	algCodes(i) = algCodeDict(ismember(algNames, algs{i}));
end

%converged = errs <= TOL;
%%%%% Convergence of noisy algos

assert(all(ismember(algNames, {'robust-l0', 'robust-l0-adaptive', 'robust-l0-trans', 'robust-l0-adaptive-trans', 'ssmp-robust', 'smp-robust'})));

mean_err1 = ms.*noise_levels*sqrt(2/pi);
sd_err1 = sqrt(ms).*noise_levels*sqrt(1 - 2/pi);
mean_signal_norm = ks*sqrt(2/pi);
upper_bound = min((mean_err1 + SOL_TOL*sd_err1)./mean_signal_norm, UPPER_BOUND_CLIP);
converged = errs_1 <= upper_bound;

%%%%%

%data = [algCodes, ks, ms, ns, converged, iters, times, errs_1, noise_levels];
%data = data(data(:, 4) > 2^20, :);
%data = data(round(log10(data(:,3)./data(:, 4))) == -3, :);

% We will plot one figure per (noise_level, delta, algorithm), one curve per (n)

%ind = ns > 2^20 & round(log10(ms./ns)) == -3;
ind = ns >= 2^22 & round(log10(ms./ns)) <= -2;

ds = [];
ds.algCodes = algCodes(ind);
ds.ks = ks(ind);
ds.ms = ms(ind);
ds.ns = ns(ind);
ds.converged = converged(ind);
ds.iters = iters(ind);
ds.times = times(ind);
ds.errs_1 = errs_1(ind);
ds.noise_levels = noise_levels(ind);

noise_level_list = unique(ds.noise_levels);

dataArray = [];

pt_data = {};

for iii = 1:length(noise_level_list)

	noise_level = noise_level_list(iii);

	ind_sigma = ds.noise_levels == noise_level;
	nList = unique(ds.ns(ind_sigma));
	nDataList = cell(1, length(nList));

	for ii = 1:length(nList)

		n = nList(ii);

		ind_n = ds.ns == n;
		mList = unique(ds.ms(ind_sigma & ind_n));

		for j = 1:length(mList)

			m = mList(j);

			ind_m = ds.ms == m; 
			algList = unique(ds.algCodes(ind_sigma & ind_m & ind_n));

			for i = 1:length(algList)

				alg = algList(i);

				ind_alg = ds.algCodes == alg; 

				if strcmp(statistic, 'mean_converged')
					indAtM = ind_alg & ind_m & ind_n & ind_sigma & ds.converged == 1;
					resultsAtM = [ds.ks(indAtM), ds.converged(indAtM), ds.times(indAtM)];
				else
					indAtM = ind_alg & ind_m & ind_n & ind_sigma;
					resultsAtM = [ds.ks(indAtM), ds.converged(indAtM), ds.times(indAtM)];
				end
				kList = sort(unique(resultsAtM(:, 1)));
				probs = zeros(size(kList));
				timings = zeros(size(kList));

				for l = 1:length(kList)
					kIndex = resultsAtM(:, 1) == kList(l);
					probs(l) = sum(resultsAtM(kIndex, 2))/length(resultsAtM(kIndex, 2));
					if strcmp(statistic, 'median')
						timings(l) = median(resultsAtM(kIndex, 3));
					else
						timings(l) = mean(resultsAtM(kIndex, 3));
					end
					%fprintf('Alg = %s, delta = %1.2f, rho = %1.2f, num_converged = %d\n', algNames{alg}, m/(ns(alg)), kList(l)/m, length(resultsAtM(kIndex, 2)));
				end
				phaseTransition = [kList/m probs timings];
				phaseTransition = phaseTransition(phaseTransition(:, 2) >= 0.5, :);
				pt_data{end+1} = {algList(i), mList(j), nList(ii), noise_level_list(iii), round(mList(j)/nList(ii), 4), phaseTransition};

			end
		end
	end
end

%%
% Plot individual probability vs time plots

plotIndividual = 0;
plotTogether = 0;
plotTimingTogether = 1;

numFigures = 0;

d = ps(1);

MS = 'MarkerSize';
LW = 'LineWidth';

colors = 'kmbgrcy';

colorList = colorscale(9, 'hue', [1/100 1], 'saturation' , 1, 'value', 0.7);
colorList(1, :) = [0 0 0];
colorList(4, :) = colorList(4, :) + [0 -0.2 0];
colorList = colorList([1 6 9 8 5 4 2 7],:);

marks = '+o*.xsd^';

wdi = './plots/';

%%
% Plot only timing aggregated by delta
% We will plot one figure per (delta, noise_level) and one curve per (algorithm, n)

if plotTimingTogether

	% We want one plot for each delta (value of m), so get a list of all m's

	alg_values = cellfun(@(v) v{1}, pt_data);
	n_values = cellfun(@(v) v{3}, pt_data);
	noise_values = cellfun(@(v) v{4}, pt_data);
	delta_values = cellfun(@(v) v{5}, pt_data);

	alg_levels = unique(alg_values);
	n_levels = unique(n_values);
	noise_levels = unique(noise_values);
	delta_levels = unique(delta_values);

	for i = 1:length(delta_levels)
		delta = delta_levels(i);
		for ii = 1:length(noise_levels)
			noise_level = noise_levels(ii);

			numFigures = numFigures + 1;
			fig = figure(numFigures);
			set(gcf, 'color', [1 1 1]);
			set(gca, 'Fontname', 'Times', 'Fontsize', 15);
			handles = [];
			labels = {};

			idx = find(delta_values == delta & noise_values == noise_level);
			for iii = 1:length(idx)
				deltaExp = round(log10(delta));
				alg_code = pt_data{idx(iii)}{1};
				alg_name = algNames{alg_code};
				n = pt_data{idx(iii)}{3};
				pt = pt_data{idx(iii)}{6};
				fprintf('Processing n = 2^{%d}, delta = 10^{%d}, sigma = %1.3f\n', log2(n), deltaExp, noise_level);

				if ~isempty(pt)
					pt
					mark_pos = find(alg_code == alg_levels);
					if n == 2^(22)
						linepattern = '--';
						linewidth = 2;
						marksize = 10;
					else
						linepattern = '-';
						linewidth = 3;
						marksize = 11;
					end
					linestyle = strcat(linepattern, s_list(mark_pos));
					handles(end + 1) = semilogy(pt(:,1), pt(:,3), linestyle, LW, linewidth, MS, marksize, 'Color', colorList(find(n == n_levels), :));
					labels{end + 1} = sprintf('n=2^{%d}, %s',  floor(log2(n)), change_names(alg_name));
					hold on;
				end

			end

			if delta == 0.001 && noise_level == 0.01
				legend(handles, labels, 'Location', 'NorthEast', 'FontSize', fs.legend);
			end

			ylim([0, 0.3*1e2]);
			xlim([0, 0.18]);
			xlabel('\rho = k/m', 'FontSize', fs.axis),...
			ylabel('Average time (sec)', 'FontSize', fs.axis);

			% Tick size
			xt = get(gca, 'XTick');
			set(gca, 'FontSize', fs.ticks);

			xt = get(gca, 'YTick');
			set(gca, 'FontSize', fs.ticks);

			title(['\delta = ', sprintf('%1.3f', delta), ', \sigma = ' sprintf('%1.3f', noise_level)], 'FontSize', fs.title);
			%title({sprintf('\delta = %1.2f', delta), ['\sigma = ' sprintf('%1.3f', noise_level)]});

			hold off;
			fileName = fullfile(wdi, ['timing_fixed_delta_single_50percent_', sprintf('%1.2f', delta), '_', sprintf('%1.3f_', noise_level), '_', statistic]);
			print('-dpdf', strcat(fileName, '.pdf'));
			print('-depsc',strcat(fileName, '.eps'));

		end
	end

end

