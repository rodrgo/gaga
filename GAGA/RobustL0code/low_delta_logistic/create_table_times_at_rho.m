
clear all;

algorithms = {'parallel_l0'};

TOL = 1e-5;
tests_per_rho = 30;

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

data = [algCodes, round(ks./ms*10000)/10000, ns, converged, iters, times];

ns = 2.^(22:2:26);
tab = [];

for i = 1:length(ns)
	average_time = mean(data(data(:, 2) > 0.04 & data(:, 2) < 0.06 & data(:, 3) == ns(i) & data(:, 4) == 1, 6));
	tab = [tab; [ns(i) average_time]];
end


tab(1, 3) = tab(2,2)/tab(1, 2);
tab(2, 3) = tab(3,2)/tab(2,2);

fprintf('\\begin{tabular}{|c|c|c|}\n');
fprintf('\\hline\n');
fprintf('$n$& time $t_{n}$ & ratio $t_{4n}/t_{n}$\\\\\n');
fprintf('\\hline\n');
fprintf('&&\\\\\n');
fprintf('$2^{22}$ & %1.4f & %1.4f\\\\\n', tab(1,2), tab(1, 3));
fprintf('&&\\\\\n');
fprintf('$2^{24}$ & %1.4f & %1.4f\\\\\n', tab(2,2), tab(2,3));
fprintf('&&\\\\\n');
fprintf('$2^{26}$ & %1.4f & - \\\\\n', tab(3,2));
fprintf('&&\\\\\n');
fprintf('\\hline\n');
fprintf('\\end{tabular}\n');

