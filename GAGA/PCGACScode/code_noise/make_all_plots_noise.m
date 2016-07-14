% make_all_plots_noise
% This script makes all plots using the lists updated here:
%   alg_list, ens_list, noise_list, vecDistr_list

%turn on (1) or off (0) ensembles to be processed.

% These are currently set to run a loop making all plots with noise and alternate 
% vector ensembles for the paper PCGACS.
dct = 0;
smv = 0;
gen = 0;


for w=1:2

  switch w
    case 1
      dct = 1; 
      smv = 1;
      gen = 0;
      n_list = [2^17];
      n_list = [2^10];  % reduction for representative data
    case 2
      dct = 0;
      smv = 0;
      gen = 1;
      n_list = [2^13];
      n_list = [2^10];  % reduction for representative data
    otherwise
      warning('all matrix ensembles turned off');
  end

%turn on (1) or off (0) the types of plots to be made
allindv = 1;		% makes individual plots for each data set
allnoise = 1;		% makes plots for each set {alg,vecdistr,matrixens} for all noise levels in noise_list
allalg = 1;		% makes plots for each set {vecdistr,matrixens,noise} for all algorithms in alg_list
alldistr = 1;		% makes plots for each set {alg,matrixens,noise} for all vector distributions vecDistr_list
allnonzeros = 1;	% makes plots for each set {alg,vecdistr,smv} for nonzero choices in nz_list 
bestalg = 1;            % makes plots for algorithm selection, fastest timing, and ratios - specifics are altered with make_all_best_alg_plots.m

% Set the list of n values to be plotted
%n_list = [2^17];

% set the noise values to be plotted
noise_list = [0 0.1];

% set the vector distributions to be plotted
vecDistr_list = [1 2 0];

% set the list of nonzeros for smv
nz_list = [4 7]; 
all_nz_list = [4 7 13];

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


if bestalg
  make_all_best_alg_plots_noise
end
