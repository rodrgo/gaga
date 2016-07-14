function make_all_plots_noise(alg_list,matens_list)
% This function makes all plots for the problem class
% (Mat,B) where the matrix ensembles Mat are defined by
% matens_list, B is the sparse binary vector distribution,
% and alg_list defines the algorithms for the plots.
% Usage:
%    make_all_plots(alg_list,matens_list)
% The input arguments are optional.
% There are no output arguments.
% The plots are stored in the directory 'plots/'.

if nargin < 2
  matens_list=cell(3,1);
  matens_list{1}='gen';
  matens_list{2}='smv';
  matens_list{3}='dct';
  if nargin < 1
    alg_list=cell(5,1);
    alg_list{1}='CGIHT';
    alg_list{2}='CGIHTrestarted';
    alg_list{1}='NIHT';
    alg_list{2}='HTP';
    alg_list{3}='CSMPSP';
  end
end

tmp=pwd;
tmp=tmp(1:end-11);
addpath([tmp '/shared']);

% This function will automatically go through the list and use the appropriate
% values for n for each ensemble.
for w=1:length(matens_list)
  ensemble=matens_list{w};
  switch ensemble
    case 'dct'
      dct = 1; 
      smv = 0;
      gen = 0;
      p = 17;
      p = 10;      % reduced for representative data
    case 'smv'
      dct = 0;
      smv = 1;
      gen = 0;
      p = 17;
      p = 10;      % reduced for representative data
    case 'gen'
      dct = 0;
      smv = 0;
      gen = 1;
      p = 13; 
      p = 10;      % reduced for representative data 
    otherwise
      warning('all matrix ensembles turned off');
  end

  n_list = 2.^p;

%turn on (1) or off (0) the types of plots to be made
allindv = 0;		% makes individual plots for each data set
allnoise = 1;		% makes plots for each set {alg,vecdistr,matrixens} for all noise levels in noise_list
allalg = 1;		% makes plots for each set {vecdistr,matrixens,noise} for all algorithms in alg_list
alldistr = 0;		% makes plots for each set {alg,matrixens,noise} for all vector distributions vecDistr_list
allnonzeros = 0;	% makes plots for each set {alg,vecdistr,smv} for nonzero choices in nz_list 
bestalg = 1;            % makes plots for algorithm selection, fastest timing, and ratios - specifics are altered with make_all_best_alg_plots.m
fixeddelta = 1;          % makes plots for fixed delta near 0.1 and 0.3; all algorithms on a semi-log plot

% Set the list of n values to be plotted if not done so above
%n_list = [2^17];

% set the noise values to be plotted
noise_list = [0 0.1 0.2];

% set the vector distributions to be plotted
vecDistr_list = [1]; %[1 2 0];

% set the list of nonzeros for smv
nz_list = [7]; 
all_nz_list = [4 7 13];

% initialization of ens_list, no need to change this as it is built
% based on which ensembles are turned on and which are turned off.
ens_list=cell(0,1);
if dct
  ens_list = [ens_list 'dct'];
end
if smv
  ens_list = [ens_list 'smv'];
end
if gen
  ens_list = [ens_list 'gen'];
end

tt=tic;

if allindv == 1;
% This makes a phase transition plot for all fixed combinations from the lists.
make_transition_plots_noise(alg_list, ens_list, noise_list, vecDistr_list);

fprintf('Completed all individual plots after %f seconds.\n',toc(tt))
end % ends if allindv

% For each value of n in n_list, we make a series of plots varying one item at a time.

for qq=1:length(n_list)
n=n_list(qq)

if allnoise == 1
% For each fixed triple of algorithm, ensemble (and nonzeros for smv), and vector distribution, 
% we make a single plot with all noise levels.
for i=1:length(alg_list)
  alg = alg_list{i};
  for j=1:length(vecDistr_list)
    vecDistribution = vecDistr_list(j);
    for kk=1:length(ens_list)
      ens = ens_list{kk};
      if ens == 'smv'
	for pp=1:length(nz_list)
	  nz=nz_list(pp);
	  make_joint_transition_plots_noise(ens, n, nz, alg, vecDistribution, noise_list);
	end
      else
	make_joint_transition_plots_noise(ens, n, 0, alg, vecDistribution, noise_list); 
      end
    end
  end
end

fprintf('Completed AllNoise plots after %f seconds.\n',toc(tt))
end % ends if allnoise


if allalg == 1;
% For each fixed triple ensemble (and nonzeros for smv), noise level, and vector distribution,
% we make a single plot with all algorithms.

for i=1:length(noise_list)
  noise_level = noise_list(i);
  for j=1:length(vecDistr_list)
    vecDistribution = vecDistr_list(j);
    for kk=1:length(ens_list)
      ens = ens_list{kk};
      if ens == 'smv'
	for pp=1:length(nz_list)
	  nz=nz_list(pp);
	  make_joint_transition_plots_algorithms(ens, n, nz, noise_level, vecDistribution, alg_list);
	end
      else
	make_joint_transition_plots_algorithms(ens, n, 0, noise_level, vecDistribution, alg_list);
      end
    end
  end
end

fprintf('Completed AllAlg plots after %f seconds.\n',toc(tt))
end % ends if allalg

if alldistr == 1;
% For each fixed triple ensemble (and nonzeros for smv), noise level, and algorithm,
% we make a single plot with all vector distributions.

for i=1:length(alg_list)
  alg = alg_list{i};
  for j=1:length(noise_list)
    noise_level = noise_list(j);
    for kk=1:length(ens_list)
      ens = ens_list{kk};
      if ens == 'smv'
	for pp=1:length(nz_list)
	  nz=nz_list(pp);
	  make_joint_transition_plots_distribution(ens, n, nz, alg, noise_level, vecDistr_list);
	end
      else
	make_joint_transition_plots_distribution(ens, n, 0, alg, noise_level, vecDistr_list);
      end
    end
  end
end

fprintf('Completed AllDistr plots after %f seconds.\n',toc(tt))
end % ends if alldist

if allnonzeros == 1;
% For each fixed set of algorith, noise_level, and vector distribution
% we plot the results using the various nonzeros for ensemble smv.
if ( (smv == 1) && (length(nz_list)>1) )
  for i=1:length(alg_list)
    alg = alg_list{i};
    for j=1:length(noise_list)
      noise_level = noise_list(j);
      for kk=1:length(vecDistr_list)
	vecDistribution=vecDistr_list(kk);
        make_joint_transition_plots_smv_nonzeros(n,all_nz_list,alg,noise_level,vecDistribution);
      end
    end
  end
end

fprintf('Completed AllNonzeros plots after %f seconds.\n',toc(tt))
end % ends if allnonzeros

end % ends n_list loop (qq)

end % ends w loop for PCGACS noise plots generation.

% make_all_best_alg_plots is a script; changes must be made on that file similar to this file
if bestalg
  make_all_best_alg_plots_noise
end

% make_fixed_delta_plots_noise is a script; changes must be made on that file similar to this file
if fixeddelta
  make_fixed_delta_plots_noise
end
