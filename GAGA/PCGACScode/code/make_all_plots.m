% make_all_plots
% This script makes all plots using the lists updated here:
%   alg_list, ens_list, noise_list, vecDistr_list

%turn on (1) or off (0) ensembles to be processed.

% These are currently set to run a loop making all plots with noise and alternate 
% vector ensembles for the paper PCGACS.
dct = 0;
smv = 0;
gen = 0;


for w=1:3

  switch w
    case 1
      dct = 1; 
      smv = 0;
      gen = 0;
      p = 10:2:20;
      p = 10;      % reduced for representative data
    case 2
      dct = 0;
      smv = 1;
      gen = 0;
      p = 10:2:18;
      p = 10;      % reduced for representative data
    case 3
      dct = 0;
      smv = 0;
      gen = 1;
      p = 10:2:14; 
      p = 10;      % reduced for representative data 
    otherwise
      warning('all matrix ensembles turned off');
  end

  n_list = 2.^p;

%turn on (1) or off (0) the types of plots to be made
allindv = 1;		% makes individual plots for each data set
allalg = 1;		% makes plots for each set {vecdistr,matrixens,noise} for all algorithms in alg_list
bestalg = 1;            % makes plots for algorithm selection, fastest timing, and ratios - specifics are altered with

% Set the list of n values to be plotted
%n_list = [2^17];

% set the noise values to be plotted
noise_list = [0];

% set the vector distributions to be plotted
vecDistr_list = [1]; 

% set the list of nonzeros for smv
nz_list = [4 7]; 

% This is the standard set of algorithms for noise tests.
% This hsould be changed if all four algorithms are not to be plotted.
alg_list=cell(3,1);
%alg_list{1}='ThresholdCG';
alg_list{1}='NIHT';
alg_list{2}='HTP';
alg_list{3}='CSMPSP';
%alg_list{4}='CGIHT';
%alg_list{2}='CGIHTrestarted';
%alg_list{5}='IHT';

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
make_transition_plots(alg_list, ens_list, noise_list, vecDistr_list);

fprintf('Completed all individual plots after %f seconds.\n',toc(tt))
end % ends if allindv



% For each value of n in n_list, we make a series of plots varying one item at a time.

for qq=1:length(n_list)
n=n_list(qq)


if allalg == 1;
% For each fixed triple ensemble (and nonzeros for smv), noise level, and vector distribution,
% we make a single plot with all algorithms.

  for j=1:length(vecDistr_list)
    vecDistribution = vecDistr_list(j);
    for kk=1:length(ens_list)
      ens = ens_list{kk};
      if ens == 'smv'
	for pp=1:length(nz_list)
	  nz=nz_list(pp);
	  make_joint_algorithms_plots(ens, n, nz, 0, vecDistribution, alg_list,1);
	end
      else
	make_joint_algorithms_plots(ens, n, 0, 0, vecDistribution, alg_list,1);
      end
    end
  end

fprintf('Completed AllAlg plots after %f seconds.\n',toc(tt))
end % ends if allalg

end % ends n_list loop (qq)

end % ends w loop for PCGACS noise plots generation.


if bestalg
  make_all_best_alg_plots
  fprintf('Completed BestAlg plots after %f seconds.\n',toc(tt))
end


