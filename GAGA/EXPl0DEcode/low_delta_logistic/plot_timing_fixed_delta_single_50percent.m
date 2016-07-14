
clear all;

algorithms = {'parallel_l0'};

TOL = 1e-5;
tests_per_rho = 30;
statistic = 'mean'; % mean, median, mean_converged

% Extract data
algs = []; ks = []; ms = []; ns = []; ds = []; errs = []; times = []; iters = [];
for i = 1:length(algorithms) 
	
	filePaths = strsplit(ls(strcat(algorithms{i}, '/gpu_data_smv*')));

	for j = 1:(length(filePaths)-1)  % carriage return gets split

		filePaths{j}
		fidi = fopen(filePaths{j});
		data = textscan(fidi,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');

		algs = [algs; strrep(strrep(data{1}, '_S_smv', ''), '_', '-')];
		ks = [ks; data{4}];
		ms = [ms; data{7}];
		ns = [ns; data{10}];
		ds = [ds; data{15}];
		errs = [errs; data{21}];
		times = [times; data{25}/1000]; % want time in seconds
		iters = [iters; data{33}];
		fclose(fidi);

	end

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
data = data(data(:, 4) > 2^20, :);
data = data(round(log10(data(:,3)./data(:, 4))) == -3, :);
size(data)

nList = unique(data(:, 4))

nDataList = cell(1, length(nList));

for ii = 1:length(nList)

	n = nList(ii);

	mList = unique(data(data(:, 4) == n, 3));

	mDataList = cell(1, length(mList));

	for j = 1:length(mList)
		algDataList = cell(1, nAlgs); % Algorithm's phase transition data at delta

		for i = 1:nAlgs
			alg = i;
			m = mList(j);
			if strcmp(statistic, 'mean_converged')
				resultsAtM = data(data(:, 1) == alg & data(:, 3) == m & data(:, 4) == n & data(:, 5) == 1, [2, 5, 7]); % take 'ks','converged' and 'times' columns
			else
				resultsAtM = data(data(:, 1) == alg & data(:, 4) == n & data(:, 3) == m, [2, 5, 7]); % take 'ks','converged' and 'times' columns
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
		mDataList{j} = algDataList;

	end
	nDataList{ii} = mDataList;
end

%%
% Plot individual probability vs time plots


plotIndividual = 0;
plotTogether = 0;
plotTimingTogether = 1;

numFigures = 0;

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

nNames = {'n = 2^{22}', 'n = 2^{24}', 'n = 2^{26}'};
log2(nList)
if plotTimingTogether
	numFigures = numFigures + 1;
	figure(numFigures);
	set(gcf, 'color', [1 1 1]);
	set(gca, 'Fontname', 'Times', 'Fontsize', 15);
	hs = [];
	numHandles = 1;
	for ii = 1:length(nList)
		n = nList(ii);
		mList = unique(data(data(:, 4) == n, 3));
		mDataList = nDataList{ii};
		fprintf('Processing n = %d\n', n);
		for j = 1:length(mList)
			algDataList = mDataList{j};
			delta = mList(j)/n
			deltaExp = round(log10(delta));
			fprintf('Processing n = 2^{%d}, delta = 10^{%d}\n', log2(n), deltaExp);

			for i = 1:length(algDataList)
				fprintf('\t%s\n', algNames{i});
				pt = algDataList{i};
				if ~isempty(pt)
					pt
					hs(numHandles) = semilogy(pt(:,1), pt(:,3), strcat('-', marks(i)), LW, 2, MS, 8, 'Color', colorList(ii, :));
					numHandles = numHandles + 1;
					ylim([0.001 10000]);
					hold on;
				end
				%hold on;
			end
		end
	end

	legend(hs, nNames, 'Location', 'NorthEast');
	xlabel('rho = k/m'),...
	ylabel('Average time (sec)');

	nExp = log2(n);
	if strcmp(statistic, 'mean')
		title(sprintf('Mean time to exact convergence for %s\nm/n = 10^{%d} and d = %d', 'parallel-l0', deltaExp, d));
		%title(sprintf('Average time to convergence\nm/n = 10^{%d} with n = 2^{%d} and d = %d', deltaExp, nExp, d));
	elseif strcmp(statistic, 'median')
		title(sprintf('Median time to convergence\nm/n = 10^{%d} with n = 2^{%d} and d = %d', deltaExp, nExp, d));
	elseif strcmp(statistic, 'mean_converged')
		title(sprintf('Mean time to exact convergence for %s\nm/n = 10^{%d} and d = %d', 'parallel-l0', deltaExp, d));
	end

	hold off;
	fileName = strcat(wdi,'timing_fixed_delta_single_50percent_', sprintf('delta_1em%d_%s', abs(deltaExp), statistic));
	print('-dpdf', strcat(fileName, '.pdf'));
	print('-depsc',strcat(fileName, '.eps'));
end

