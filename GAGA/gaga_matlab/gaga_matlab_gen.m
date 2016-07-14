function [errors, timings, iter, checkSupport, convRate, xout] = gaga_matlab_gen(algstr, k, m, n, options)
% This function has the usage 
% [errors, timings, iter, checkSupport, convRate, xout] = gaga_matlab_gen(algstr, k, m, n, options);
% algstr is string denoting which algorithm should be used to solve the problem:
%   ('NIHT', 'IHT', 'HTP', 'CSMPSP', 'ThresholdSD', 'ThresholdCG'),
% k is the assumed sparsity, 
% m is the number of measurements, 
% n is the dimension of the signal one attempts to recover,
% options are generated using gagaOptions (help gagaOptions shows usage)
% The outputs are:
%  errors= [l2 error, l1 error, l-infity error]
%  timings= [algorithm time, average iteration time, total test time]
%  iter= number of iterations in the algorithm
%  checkSupport= [TruePositive, FalsePositive, TrueNegative, FalseNegative]
%  convRate= convergence rate of the algorithm
%  xout= k-sparse solution vector
% Additional output is written to a date stamped text file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% One difference in the matlab code is that we never employ the idea of
% only counting a fraction of the bins.  This is because the counting must
% be done in serial anyway, so there is no advantage in counting only a
% subset of the bins.  The advantage in GAGA comes from reducing the number
% of conflicts with atomicAdd by not couting the bins containing the
% smallest entries.
% A second difference is that one can not overload a matlab functon
% in a straightforward manner.  Thus, if one wishes to pass a problem to an
% algorithm directly, they should do so using the algorithm rather than this 
% parent function.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jeffrey D. Blanchard and Jared Tanner, 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following replicates #include "greedyheader.cu".  You must change the
% path in greedyheader.m to the appropriate location for GAGA_matlab.
greedyheader


timeTest = tic;

% determine the algorithm from the input string
valid_alg=0;
if ( strcmp(algstr, 'NIHT') || strcmp(algstr, 'HTP') || strcmp(algstr, 'IHT') || strcmp(algstr, 'ThresholdSD') || strcmp(algstr, 'ThresholdCG') || strcmp(algstr, 'CSMPSP') || strcmp(algstr, 'CGIHT') )
  valid_alg = 1;
end


if valid_alg == 0 
	fprintf('[gaga_matlab_gen Error: The possible (case sensitive) input strings for algorithms using gaga_matlab_gen are: \n NIHT\n IHT\n HTP\n ThresholdSD\n ThresholdCG\n CSMPSP\n CGIHT\n');
else

% initialise options at default values
% some of these may not be used depending on the usage (such as vecDistribution)
vecDistribution = 1; % 'binary'  
matrixEnsemble = 1; %'gaussian';  
seed = sum(100*clock);
tol = 10^(-4);
noise_level = 0.0;
timingFlag = 0; % off
restartFlag = 0; % off
supp_flag = 0; % dynamic binning
if ( strcmp(algstr,'HTP') || strcmp(algstr,'CSMPSP') ) 
  maxiter=300;
else 
  maxiter=5000;
end

% extract the options if the last argument is a cell
if (nargin == 5) 
  if (iscell(options) == 0)
    error('options input should be a cell, use gagaOptions');
  else
    numOptions = size(options,1);
    for i=1:numOptions
      % go through the list of possible options looking for a match 

      if (strcmp(options{i,1}, 'tol') == 1)
	tol = options{i,2};
      elseif (strcmp(options{i,1}, 'vecDistribution') == 1)
	if (strcmp(options{i,2},'binary')) vecDistribution = 1; 
        elseif (strcmp(options{i,2},'uniform')) vecDistribution = 0;
        elseif (strcmp(options{i,2},'gaussian')) vecDistribution = 2;
        end
      elseif (strcmp(options{i,1}, 'matrixEnsemble') == 1)
	if (strcmp(options{i,2},'binary')) matrixEnsemble = 2; 
        elseif (strcmp(options{i,2},'gaussian')) matrixEnsemble = 1;
        end
      elseif (strcmp(options{i,1}, 'seed') == 1)
	seed = options{i,2};
      elseif (strcmp(options{i,1}, 'supportFlag') == 1)
	supp_flag = options{i,2};
      elseif (strcmp(options{i,1}, 'noise') == 1)
	noise_level = options{i,2};
      elseif (strcmp(options{i,1}, 'timing') == 1)
	if (strcmp(options{i,2},'on')) timingFlag = 1; 
        elseif (strcmp(options{i,2},'off')) timingFlag = 0;
        end
      elseif (strcmp(options{i,1}, 'restartFlag') == 1)
	if (strcmp(options{i,2},'on')) restartFlag = 1; 
        elseif (strcmp(options{i,2},'off')) restartFlag = 0;
        end
      elseif (strcmp(options{i,1}, 'maxiter') == 1)
	maxiter = options{i,2};
      else
	display(sprintf('The following option is not recognised: %s.',options{i,1}))
      end
    end
  end
end


% create a random problem
if (noise_level <= 0)
  [vec_input, y, A_gen, seed] = createProblem_gen(k, m, n, vecDistribution, matrixEnsemble, seed);
else
  [vec_input, y, A_gen, seed] = createProblem_gen_noise(k, m, n, vecDistribution, matrixEnsemble, seed, noise_level);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the problem with the input algorithm %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

timetotal = tic;

% Initialization of parameters

iter = 0;
ksum = 0;  % In GAGA this is called sum, but in matlab 'sum' is a built-in function.

time_sum = 0;

if (strcmp(algstr, 'NIHT')==1) alg = 0;
elseif (strcmp(algstr, 'HTP')==1) alg = 1;
elseif (strcmp(algstr, 'IHT')==1) alg = 2;
elseif (strcmp(algstr, 'ThresholdSD')==1) alg = 3;
elseif (strcmp(algstr, 'ThresholdCG')==1) alg = 4;
elseif (strcmp(algstr, 'CSMPSP')==1) alg = 5;
elseif (strcmp(algstr, 'CGIHT')==1) alg = 6;
end

switch alg;
  case 0
    if (timingFlag == 0)
	[vec, residNorm_prev, iter, time_sum, ksum] = NIHT_M_gen(y, A_gen, k, m, n, tol, maxiter);
    else
        [vec, residNorm_prev, iter, time_sum, ksum, time_per_iteration, time_supp_set] = NIHT_M_timings_gen(y, A_gen, k, m, n, tol, maxiter, supp_flag);
    end
  case 1
    if (timingFlag == 0)
	[vec, residNorm_prev, iter, time_sum, ksum] = HTP_M_gen(y, A_gen, k, m, n, tol, maxiter);
    else
        [vec, residNorm_prev, iter, time_sum, ksum, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg] = HTP_M_timings_gen(y, A_gen, k, m, n, tol, maxiter, supp_flag);
    end
  case 2
	[vec, residNorm_prev, iter, time_sum, ksum] = IHT_M_gen(y, A_gen, k, m, n, tol, maxiter);
  case 3
	[vec, residNorm_prev, iter, time_sum, ksum] = HT_SD_M_gen(y, A_gen, k, m, n, tol, maxiter);
  case 4
	[vec, residNorm_prev, iter, time_sum, ksum] = HT_CG_M_gen(y, A_gen, k, m, n, tol, maxiter);
  case 5
	[vec, residNorm_prev, iter, time_sum, ksum] = CSMPSP_M_gen(y, A_gen, k, m, n, tol, maxiter);
  case 6
	[vec, residNorm_prev, iter, time_sum, ksum] = CGIHT_M_gen(y, A_gen, k, m, n, tol, maxiter, restartFlag);
  otherwise fprintf('[gaga_matlab_gen Error: The possible (case sensitive) input strings for algorithms using gaga_matlab_gen are: \n NIHT\n IHT\n HTP\n ThresholdSD\n ThresholdCG\n CSMPSP\n CGIHT\n');
end

xout=vec;


totaltime=toc(timetotal);

%%%%%%%%%%%%%%%%%%%%%
% Check the Results %
%%%%%%%%%%%%%%%%%%%%%

if ( (strcmp(algstr, 'CGIHT')==1) && (restartFlag==1) )
  algstr = strcat(algstr, 'restarted');
end

checkSupport = [0; 0; 0; 0];

if (noise_level <= 0)
  if (timingFlag == 1)
    if (alg == 0)
      [errors, timings, checkSupport, convRate] = results_timings_gen(vec, vec_input, ...
        vecDistribution, residNorm_prev, iter, checkSupport, ...
        totaltime, time_sum, time_per_iteration, time_supp_set, ...
        ksum, 0, supp_flag, k, m, n, matrixEnsemble, seed, timeTest, algstr);
    elseif (alg == 1)
      [errors, timings, checkSupport, convRate] = results_timings_HTP_gen(vec, vec_input, ...
        vecDistribution, residNorm_prev, iter, checkSupport, ...
        totaltime, time_sum, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, ...
        ksum, 0, supp_flag, k, m, n, matrixEnsemble, seed, timeTest, algstr);
    else
      display('Error: timings flag only valid for algorithms: NIHT and HTP');
    end
  else
    [errors, timings, checkSupport, convRate] = results_gen(vec, vec_input, ...
        vecDistribution, residNorm_prev, iter, checkSupport, ...
        totaltime, time_sum, ksum, k, m, n, matrixEnsemble, seed, timeTest, algstr);
  end
else
  if (timingFlag == 1)
    display('Error: timings flag only valid if noise is set to zero.');
  else
    [errors, timings, checkSupport, convRate] = results_gen_noise(vec, vec_input, ...
        vecDistribution, residNorm_prev, iter, checkSupport, ...
        totaltime, time_sum, ksum, k, m, n, matrixEnsemble, seed, noise_level, timeTest, algstr);
  end
end


end % closes the else ensuring the algorithm input was valid    
    
