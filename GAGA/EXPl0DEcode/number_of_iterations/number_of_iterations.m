
run('init.m');
wdi = './plots/';

LW = 'LineWidth';
MS = 'MarkerSize';

numColors = 6;
colorList = colorscale(numColors, 'hue', [1/(numColors + 1) 1], 'saturation' , 0.9, 'value', 0.9);

colorList = colorscale(9, 'hue', [1/100 1], 'saturation' , 1, 'value', 0.7);
colorList(1, :) = [0 0 0];
colorList(4, :) = colorList(4, :) + [0 -0.2 0];
colorList = colorList([1 6 9 8 5 4 2 7],:);

markerList = '+o*.xsdh^v><p';
names_list = {'plus', 'circle', 'asterisk', 'point', 'cross', 'square', 'diamond', 'hexagram', 'up-triangle', 'down-triangle', 'right-triangle', 'left-triangle', 'pentagram'};

%%%
%%% Convergence for fixed delta
%%%

logN = 20;
n = 2^logN;
delta = 0.1;
d = 7;

TOL = 1e-5;
seed = 100;

gpuNumber = 0;

% What should we test?

classes = {'linear', 'sublinear'};

algSets{1} = {'er', 'ssmp'};
maxiterSets{1} = {'3k', '3k'};

algSets{2} = {'parallel_l0', 'parallel_lddsr', 'serial_l0'};
maxiterSets{2} = {'1k', '1k', '1k'};

% Should we plot?

plotIt = 1;

% Run tests

for cc = 1:length(classes)
	algs = algSets{cc};
	maxiters = maxiterSets{cc};

	m = ceil(delta*n);

	rhos = 0.005:0.005:0.1;
	iterations = {};
	for i = 1:length(rhos)
		iterations{i} = zeros(size(rhos));
	end

	numFigures = 0;
	for l = 1:length(rhos)
		rho = rhos(l);
		k = ceil(rho*m);

		seed = seed + 11;
		names = {};

		for j = 1:length(algs)
			if strcmp(maxiters{j}(2), 'k')
				MAXITER = str2num(maxiters{j}(1))*k;
			elseif strcmp(maxiters{j}(2), 'n')
				MAXITER = str2num(maxiters{j}(1))*n;
			end
			options = gagaOptions('gpuNumber', gpuNumber, 'maxiter', MAXITER,'tol', TOL,'noise',single(0.0),'vecDistribution', 'gaussian', 'seed', seed, 'kFixed','on');
			[errors times iters supp conv xhat resRecord timeRecord] = gaga_cs(algs{j}, 'smv', int32(k), int32(m), int32(n), int32(d), options);
			iterations{j}(l) = iters;
			names{end + 1} = strrep(algs{j}, '_', '-');
		end

		fprintf('rho = %1.2f, done.\n', rhos(l));
	end

	if plotIt
		numFigures = numFigures + 1;
		figure(numFigures);
		set(gcf, 'color', [1 1 1]);
		set(gca, 'Fontname', 'Times', 'Fontsize', 15);
		
		handles = [];
		
		for i = 1:length(names)
			handles(i) = plot(rhos, (iterations{i}), strcat('-', markerList(i)), 'Color', colorList(i, :), LW, 1.5, MS, 10);
			hold on;
		end
		if (strcmp(classes{cc}, 'linear') == 1) 
			handles(end + 1) = plot(rhos, ((rhos*m) + 500), strcat('--'), 'Color', colorList(length(handles) + 1, :), LW, 1.5, MS, 10);
			hold on;
			names{end + 1} = 'k + C';
			legend(handles, names, 'Location', 'NorthWest');
		else 
			handles(end + 1) = plot(rhos, (log(rhos*m)), strcat('--'), 'Color', colorList(length(handles) + 1, :), LW, 1.5, MS, 10);
			hold on;
			names{end + 1} = 'log(k)';
			legend(handles, names, 'Location', 'NorthWest');
		end
		title(sprintf('Iterations for convergence\nm/n = %1.2f with n = 2^{%d} and d = %d', delta, logN, d));
		
		fileName = strcat(wdi,'iterations_vs_rho_', sprintf('%s', classes{cc}) ,'_delta_', sprintf('%1.2f', delta));
		print('-dpdf', strcat(fileName, '.pdf'));
		print('-depsc',strcat(fileName, '.eps'));
		hold off;
	end

end
