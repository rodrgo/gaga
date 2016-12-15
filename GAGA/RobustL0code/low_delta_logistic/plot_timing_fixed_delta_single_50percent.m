
clear all;

algorithms = {'deterministic_robust_l0'};
statistic = 'mean'; % mean, median, mean_converged

% ===============
% Set tolerances
% ==============

SOL_TOL = 1;
RES_TOL = 1;

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

assert(all(ismember(algNames, {'deterministic-robust-l0', 'robust-l0', 'ssmp-robust'})));

mean_err1 = ms.*noise_levels*sqrt(2/pi);
sd_err1 = sqrt(ms).*noise_levels*sqrt(1 - 2/pi);
mean_signal_norm = ks*sqrt(2/pi);
converged = errs_1 <= (mean_err1 + SOL_TOL*sd_err1)./mean_signal_norm;

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
noiseDataList = cell(1, length(noise_level_list));

dataArray = [];

for iii = 1:length(noise_level_list)

	noise_level = noise_level_list(iii);

	ind_sigma = ds.noise_levels == noise_level;
	nList = unique(ds.ns(ind_sigma));
	nDataList = cell(1, length(nList));

	for ii = 1:length(nList)

		n = nList(ii);

		ind_n = ds.ns == n;
		mList = unique(ds.ms(ind_sigma & ind_n));
		mDataList = cell(1, length(mList));

		for j = 1:length(mList)

			m = mList(j);

			ind_m = ds.ms == m; 
			algList = unique(ds.algCodes(ind_sigma & ind_m & ind_n));
			algDataList = cell(1, length(algList));

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
				algDataList{i} = phaseTransition;

			end
			mDataList{j} = {algList, algDataList};

		end
		nDataList{ii} = {mList, mDataList};
	end
	noiseDataList{iii} = {nList, nDataList};
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
% We will plot one figure per (noise_level) and one curve per (delta, algorithm, n)

if plotTimingTogether

	for iii = 1:length(noise_level_list)

		numFigures = numFigures + 1;
		figure(numFigures);
		set(gcf, 'color', [1 1 1]);
		set(gca, 'Fontname', 'Times', 'Fontsize', 15);
		handles = [];
		labels = [];

		noise_level = noise_level_list(iii);
		nList = noiseDataList{iii}{1};
		nDataList = noiseDataList{iii}{2};

		for ii = 1:length(nList)

			n = nList(ii);
			mList = nDataList{ii}{1};
			mDataList = nDataList{ii}{2};

			for j = 1:length(mList)
				algList = mDataList{j}{1};
				algDataList = mDataList{j}{2};

				delta = mList(j)/n
				deltaExp = round(log10(delta));
				fprintf('Processing n = 2^{%d}, delta = 10^{%d}, sigma = %1.3f\n', log2(n), deltaExp, noise_level);

				for i = 1:length(algDataList)
					fprintf('\t%s\n', algNames{i});
					pt = algDataList{i};
					if ~isempty(pt)
						pt
						handles(end + 1) = semilogy(pt(:,1), pt(:,3), strcat('-', marks(j)), LW, 2, MS, 8, 'Color', colorList(ii, :));
						labels{end + 1} = ['\delta=' sprintf('%1.3f, n=2^{%s}', delta, num2str(log2(n)))];
						hold on;
					end
					%hold on;
				end
			end
		end

		legend(handles, labels, 'Location', 'NorthWest');
		ylim([0, 1e2]);
		xlim([0, 0.2]);
		xlabel('\rho = k/m'),...
		ylabel('Average time (sec)');

		title({'deterministic-robust-l0', ['\sigma = ' sprintf('%1.3f', noise_level)]});

		hold off;
		fileName = fullfile(wdi, ['timing_fixed_delta_single_50percent_', sprintf('%1.3f_', noise_level_list(iii)), '_', statistic]);
		print('-dpdf', strcat(fileName, '.pdf'));
		print('-depsc',strcat(fileName, '.eps'));
	end
end

