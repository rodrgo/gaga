function generate_data_timings(alg_list,matens_list)
% This function generates gpu data for determining algorithm selection maps 
% for the problem class (Mat,B) with no noise and for all algorithms in alg_list.
% Optinal Inputs:
%  alg_list is a cell with strings listing the desired algorithms
%  matens_list is a cell with strings listing the desired matrix ensembles

% 1. add the path to GAGA when using this script individually
%temp=pwd;
%GAGApath=temp(1:end-16);
%addpath([GAGApath '/gaga']);
%addpath([GAGApath '/shared']);

% 2. set the GPU number to use one of multiple gpus
gpuNum = 0; % default should be 0

% 3. set the delta_list through the phase space
delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
delta_list=sort(delta_list);
delta_list=[0.05 0.1 0.2 0.3 0.5 0.7];  % reduction for representative data; comment out for full data.

rho_list=0.02:0.02:0.98;

% 4. set the values of n you wish to test
n_list_dct=2^10;      % this smaller value of n is for quickly generating representative data from PCGACS
n_list_smv=2^10;      % this smaller value of n is for quickly generating representative data from PCGACS
n_list_gen=2^10;      % this smaller value of n is for quickly generating representative data from PCGACS
% n_list_dct=[2^10 2^12 2^14 2^16 2^18 2^20];  % uncomment this line to generate the data for PCGACS
% n_list_smv=[2^10 2^12 2^14 2^16 2^18];       % uncomment this line to generate the data for PCGACS
% n_list_gen=[2^10 2^12 2^14];                 % uncomment this line to generate the data for PCGACS

% 5. set all vector distributions with three total options: 'binary' (default), 'guassian', and 'uniform'.
vecDistr_list=cell(1,1);
vecDistr_list{1}='binary';
%vecDistr_list{2}='gaussian';
%vecDistr_list{3}='uniform';

% 6. set the matrix ensembles you wish to test.  First choose from 'dct','smv','gen' and then 
% set the distribution of values in 'gen' or 'smv'
if nargin<2
  matens_list = cell(3,1);
  matens_list{1} = 'dct';
  matens_list{2} = 'smv';
  matens_list{3} = 'gen';
end

matrixEnsemble_gen = 'gaussian'; % for gen the options are 'gaussian' (default) or 'binary'.
matrixEnsemble_smv = 'binary';   % for smv the options are 'binary (default) or 'ones'.

% 7.  If testing the 'smv' matrix ensembel, set all desired choices for the number of nonzeros per column.
nonzeros_list=[4 7];

% 8.  Set the algorithms you want to test.
if nargin<1
  alg_list=cell(3,1);
  alg_list{1}='NIHT';
  alg_list{2}='HTP';
  alg_list{3}='CSMPSP';
end

% 9.  Set all noise levels you wish to test where the noise levels are a scaling factor for ||y||=||Ax||.
noise_list = [0];

% 10.  Set additional parameters for testing.
%use 5000 for NIHT and 100 or 300 for HTP
maxiter_niht=500; 
maxiter_htp=300;
Tol=10^(-4);
tests_per_k=5;  

m_minimum=2^3;

%%% Option 11 is not relevant for PCGACS which did not include CGIHT
%%% This script is appropriate for GAGA 1.1.0 which includes CGIHT
%%% For PCGACS data, leave this entry as is.
% 11. set the restart options for CGIHT
% restart_list gives the options 'on' and 'off' for CGIHT.  
% To test only one version of CGIHT, alter the length of the list.
restart_list=cell(1,1);
restart_list{1}='on';
%restart_list{2}='off';


% 12.  set the definition of success in terms of l_infty or l_two
% DEFAULT: l_infty_success uses    success = success + (en(end)<tol*10)
% which measures the relative l_infty norm against the tolerance.
% the l_two_success uses      success = success + (en(2) < .001 + 2*noise_level)
% which measures the relative l_two norm against a fixed criterion including the noise.
l_two_success = 0; % 0 off uses l_infty, 1 on uses l_two


%%%%%%% If multiple GPUs are available, this can divide the work

if gpuNum==2
delta_list=delta_list(1:3:end);
elseif gpuNum==4
delta_list=delta_list(2:3:end);
elseif gpuNum==6
delta_list=delta_list(3:3:end);
else
delta_list=delta_list;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The following code runs all tests for the options you have chosen in 1.-11. %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttt=tic;

for me=1:length(matens_list)
  matens = matens_list{me};
  if strcmp(matens,'smv')
    matrixEnsemble = matrixEnsemble_smv;
    nz_list = nonzeros_list;
    n_list = n_list_smv;
  elseif strcmp(matens,'gen')
    matrixEnsemble = matrixEnsemble_gen;
    nz_list = 0;    % this is not needed for dct or gen, so the nz_list loop will be length 1.
    n_list = n_list_gen;
  elseif strcmp(matens,'dct')
    nz_list = 0;    % this is not needed for dct or gen, so the nz_list loop will be length 1.
    n_list = n_list_dct;
  else
    error('Invlaid matrix ensemble in matens_list.');
  end

for zz=1:length(nz_list);
  nz = nz_list(zz);

for vd=1:length(vecDistr_list)
  vecDistribution = vecDistr_list{vd};
  

for ii=1:length(n_list)
  n=n_list(ii);
  if ( strcmp(matens,'gen') && ( n>2^14 ) )
    n=2^13;
    warning('For matrix ensemble gen, n>2^14, reset to n=2^13.');
  end

  for hh=1:length(alg_list)
    alg=alg_list{hh};
    if ( strcmp(alg,'HTP') || strcmp(alg,'CSMPSP') )
      maxIter=maxiter_htp;  % using a projection based algorithm
    else 
      maxIter=maxiter_niht; % using a non-projection based algorithm
    end

%%%%%%%%%%%%%%%%   SET OPTIONS and run tests  %%%%%%%%%%%%%%%%%%%%

for nl=1:length(noise_list)
  noise_level = noise_list(nl);
  if strcmp(alg,'CGIHT')
    for restart=1:length(restart_list)
      if strcmp(matens,'dct')
        options=gagaOptions('tol',Tol,'maxiter',maxIter,'vecDistribution',vecDistribution,...
	                       'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag',restart_list{restart});
      else
        options=gagaOptions('tol',Tol,'maxiter',maxIter,'vecDistribution',vecDistribution,'matrixEnsemble',matrixEnsemble,...
	                       'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag',restart_list{restart});
      end



    for jj=1:length(delta_list);
      m=ceil(n*delta_list(jj)); 
      
      if m>=m_minimum
        k_list=ceil(m*rho_list);
        k_list=max(k_list,1); k_list=min(k_list,m-1);
        k_list=intersect(k_list,k_list);
        for ll=1:length(k_list)
          k=k_list(ll);
          success=0;
          
          for p=1:tests_per_k
            if strcmp(matens,'smv')
              [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, nz, options);
            else
              [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, options);
            end
            if l_two_success
	      success = success + ( en(2) < .001+2*noise_level );
            else
              success=success+(en(end)<(Tol*10));
            end
          end
          if success==0
            break
          end
          
        end %ends k_list loop
      end %ends if m>m_minimum  

    display(sprintf('%s with %s and restarting=%s for n=%d, m=%d, k_max=%d with noise level %0.2f finished in %0.2f seconds',alg,matens,restart_list{restart},n,m,k,noise_level,toc(ttt)));

    end % ends the delta_list loop for the value of m

    end % ends the restart loop
  else  % the else is for determining if the algorithm is CGIHT which uses restarting options
    if strcmp(matens,'dct')
      options=gagaOptions('tol',Tol,'maxiter',maxIter,'vecDistribution',vecDistribution,...
                      'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on');
    else
      options=gagaOptions('tol',Tol,'maxiter',maxIter,'vecDistribution',vecDistribution,'matrixEnsemble',matrixEnsemble,...
                      'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on');
    end



    for jj=1:length(delta_list);
      m=ceil(n*delta_list(jj)); 
      
      if m>=m_minimum
        k_list=ceil(m*rho_list);
        k_list=max(k_list,1); k_list=min(k_list,m-1);
        k_list=intersect(k_list,k_list);
        for ll=1:length(k_list)
          k=k_list(ll);
          success=0;
          
          for p=1:tests_per_k
            if strcmp(matens,'smv')
              [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, nz, options);
            else
              [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, options);
            end
            if l_two_success
	      success = success + ( en(2) < .001+2*noise_level );
            else
              success=success+(en(end)<(Tol*10));
            end
          end
          if success==0
            break
          end
          
        end %ends k_list loop
      end %ends if m>m_minimum  

    display(sprintf('%s with %s for n=%d, m=%d, k_max=%d with noise level %0.2f finished in %0.2f seconds',alg,matens,n,m,k,noise_level,toc(ttt)));

    end % ends the delta_list loop for the value of m

  end % ends if strcmp(alg,'CGIHT')
end % ends noise_list loop


  end %ends alg_list loop
end % ends n_list loop

end % ends vecDistr_list loop
end % ends nz_list loop for the nozeros_list with smv
end % ends matens_list loop

