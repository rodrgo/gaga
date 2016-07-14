
clear all;

algorithms = {'smp', 'ssmp', 'parallel_lddsr', 'er', 'parallel_l0', 'serial_l0'};

TOL = 1e-4;
tests_per_rho = 10;
statistic = 'mean_converged'; % mean, median, mean_converged

% Extract data
algs = []; ks = []; ms = []; ns = []; ds = []; errs = []; times = []; iters = [];
for i = 1:length(algorithms) 

	fidi = fopen(strcat(algorithms{i}, '/', 'gpu_data_smv20150302.txt'));
	data = textscan(fidi,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');

	algs = [algs; strrep(strrep(data{1}, '_S_smv', ''), '_', '-')];
	ks = [ks; data{4}];
	ms = [ms; data{7}];
	ns = [ns; data{10}];
	ds = [ds; data{15}];
	errs = [errs; data{21}];
	times = [times; data{25}/1000]; % want time in seconds
	iters = [iters; data{33}];

end

% Create dictionary of names
algNames = unique(algs);
nAlgs = length(algNames);
algCodeDict = 1:nAlgs;
algCodes = zeros(size(algs));
for i = 1:length(algs)
	algCodes(i) = algCodeDict(ismember(algNames, algs{i}));
end

converged = errs <= TOL;

data = [algCodes, ks, ms, ns, converged, iters, times];

mList = unique(ms);
mListNum = length(mList);

algDataList = cell(1, nAlgs);

for i = 1:nAlgs
	algPT = cell(1, mListNum); % Algorithm's phase transition data at delta
	for j = 1:mListNum
		alg = i;
		m = mList(j);
		if strcmp(statistic, 'mean_converged')
			resultsAtM = data(algCodes == alg & ms == m & converged == 1, [2, 5, 7]); % take 'ks','converged' and 'times' columns
		else
			resultsAtM = data(algCodes == alg & ms == m, [2, 5, 7]); % take 'ks','converged' and 'times' columns
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
		algPT{j} = phaseTransition;
	end
	algDataList{i} = algPT;
end

%%
% Plot individual probability vs time plots


plotIndividual = 0;
plotTogether = 0;
plotTimingTogether = 1;

numFigures = 0;

n = ns(1);
d = ds(1);

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

if plotTimingTogether
	for j = 1:length(mList)
		delta = mList(j)/n;
		fprintf('Processing delta = %1.2f\n', delta);
		hs = [];

		numFigures = numFigures + 1;
		figure(numFigures);
		set(gcf, 'color', [1 1 1]);
		set(gca, 'Fontname', 'Times', 'Fontsize', 15);
		for i = 1:length(algDataList)
			fprintf('\t%s\n', algNames{i});
			algData = algDataList{i};
			pt = algData{j}; % phase Transition
			hs(i) = semilogy(pt(:,1), pt(:,3), strcat('-', marks(i)), LW, 2, MS, 8, 'Color', colorList(i, :));
			ylim([0.001 10000]);
			hold on;
			%hold on;
		end
		legend(hs, algNames, 'Location', 'NorthEast');
		xlabel('rho = k/m'),...
		ylabel('Average time (sec)');

		nExp = log2(n);
		if strcmp(statistic, 'mean')
			title(sprintf('Average time to convergence\nm/n = %1.2f with n = 2^{%d} and d = %d', delta, nExp, d));
		elseif strcmp(statistic, 'median')
			title(sprintf('Median time to convergence\nm/n = %1.2f with n = 2^{%d} and d = %d', delta, nExp, d));
		elseif strcmp(statistic, 'mean_converged')
			title(sprintf('Mean time to exact convergence\nm/n = %1.2f with n = 2^{%d} and d = %d', delta, nExp, d));
		end

		hold off;
		fileName = strcat(wdi,'timing_fixed_delta_ALL_', sprintf('delta_%1.2f_%s', delta, statistic));
		print('-dpdf', strcat(fileName, '.pdf'));
		print('-depsc',strcat(fileName, '.eps'));
	end
end

