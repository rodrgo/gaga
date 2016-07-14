%this script generates gpu data for any matrix ensemble, vector distribution,
%noise level, algorithm, etc. 

% 1. when using this file by itself, you might need to add paths to GAGA and the location of tests_delta_fixed.m
%temp=pwd;
%GAGApath=temp(1:end-22);
%addpath([GAGApath '/gaga']);
%addpath([GAGApath '/shared']);

% 2. set the GPU number to use one of multiple gpus
gpuNum = 0; % default should be 0

% 3. set the kmn_list for the tables directly
kmn_list=cell(3,1);

% kmn_list for dct.  Ensure matens_list{1} = 'dct'
k_list=[141 211 253 1239 1858 2229 2401 3601 4321 10451 15676 18811 37721 56581 67897];
m_list=[2622*ones(1,3) 15729*ones(1,3) 26215*ones(1,3) 75332*ones(1,3) 161288*ones(1,3)];
n_list=262144*ones(1,15);
kmn_list{1}=[k_list' m_list' n_list'];
kmn_list{1}=[k_list(1:6)' m_list(1:6)' n_list(1:6)'];  % reduction for representative data

% kmn_list for smv.  Ensure matens_list{2} = 'smv'
k_list=[32 48 58 305 458 549 589 883 1060 2460 3690 4428 7656 11484 13780];
m_list=[656*ones(1,3) 3933*ones(1,3) 6554*ones(1,3) 18833*ones(1,3) 40322*ones(1,3)];
n_list=65536*ones(1,15);
kmn_list{2}=[k_list' m_list' n_list'];
kmn_list{2}=[k_list(1:6)' m_list(1:6)' n_list(1:6)'];  % reduction for representative data

% kmn_list for gen.  Ensure matens_list{3} = 'gen'
k_list=[19 28 34 37 56 67 153 230 275 477 716 859];
m_list=[246*ones(1,3) 410*ones(1,3) 1178*ones(1,3) 2521*ones(1,3)];
n_list=4096*ones(1,12);
kmn_list{3}=[k_list' m_list' n_list'];
kmn_list{3}=[k_list(1:6)' m_list(1:6)' n_list(1:6)'];  % reduction for representative data


% 4. set all vector distributions with three total options: 'binary' (default), 'guassian', and 'uniform'.
vecDistr_list=cell(1,1);
vecDistr_list{1}='binary';
%vecDistr_list{2}='gaussian';
%vecDistr_list{3}='uniform';

% 5. set the matrix ensembles you wish to test.  First choose from 'dct','smv','gen' and then 
% set the distribution of values in 'gen' or 'smv'
if nargin<2
  matens_list = cell(3,1);
  matens_list{1} = 'dct';
  matens_list{2} = 'smv';
  matens_list{3} = 'gen';
end

matrixEnsemble_gen = 'gaussian'; % for gen the options are 'gaussian' (default) or 'binary'.
matrixEnsemble_smv = 'binary';   % for smv the options are 'binary (default) or 'ones'.

% 6.  If testing the 'smv' matrix ensembel, set all desired choices for the number of nonzeros per column.
nonzeros_list=[7]; % [4 7 13]

% 7.  Set the algorithms you want to test.
if nargin<1
  alg_list=cell(3,1);
  alg_list{1}='NIHT';
  alg_list{2}='HTP';
  alg_list{3}='CSMPSP';
end


% 8.  Set all noise levels you wish to test where the noise levels are a scaling factor for ||y||=||Ax||.
noise_list = [0 0.1]

% 10.  Set additional parameters for testing.
%use 5000 for NIHT and 100 or 300 for HTP
maxiter_niht=5000; 
maxiter_htp=300;
Tol=10^(-4);
tests_per_k=120;  

m_minimum=2^3;

%%%%%%%%%%%%%  Not needed for PCGACS data %%%%%%%%%%%%%%%
%%%%%%%%%%%%%  for PCGACS data, leave this as is %%%%%%%%
% 11. set the restart options for CGIHT
% restart_list gives the options 'on' and 'off' for CGIHT.  
% To test only one version of CGIHT, alter the length of the list.
restart_list=cell(1,1);
restart_list{1}='on';
%restart_list{2}='off';


%%%%%%%%%%%%%  Not needed for PCGACS data %%%%%%%%%%%%%%%
%%%%%%%%%%%%%  for PCGACS data, leave this as is %%%%%%%%
% 12.  set the definition of success in terms of l_infty or l_two
% DEFAULT: l_infty_success uses    success = success + (en(end)<tol*10)
% which measures the relative l_infty norm against the tolerance.
% the l_two_success uses      success = success + (en(2) < .001 + 2*noise_level)
% which measures the relative l_two norm against a fixed criterion including the noise.
l_two_success = 1; % 0 off uses l_infty, 1 on uses l_two

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The following code runs all tests for the options you have chosen in 1.-11. %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttt=tic;

for me=1:length(matens_list)
  matens = matens_list{me};
  if strcmp(matens,'smv')
    matrixEnsemble = matrixEnsemble_smv;
    nz_list = nonzeros_list;
  else
    matrixEnsemble = matrixEnsemble_gen;
    nz_list = nonzeros_list(1);    % this is not needed for dct or gen, so the nz_list loop will be length 1.
  end

for zz=1:length(nz_list);
  nz = nz_list(zz);

for vd=1:length(vecDistr_list)
  vecDistribution = vecDistr_list{vd};
  

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




    for ii=1:length(kmn_list{me})

      k=kmn_list{me}(ii,1);
      m=kmn_list{me}(ii,2);
      n=kmn_list{me}(ii,3);

      for p=1:tests_per_k
        if strcmp(matens,'smv')
          [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, nz, options);
        else
          [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, options);
        end
      end % ends the tests_per_k loop

      if strcmp(matens,'smv')
        display(sprintf('%s restarting %s %s with %d nonzeros per column, n=%d, m=%d, and k=%d, completed after %f seconds.',alg,restart_list{restart},matens,nz,n,m,k,toc(ttt)))
      else
        display(sprintf('%s restarting %s %s, n=%d, m=%d, and k=%d, completed after %f seconds.',alg,restart_list{restart},matens,n,m,k,toc(ttt)))
      end
    end % ends the kmn_list{me} loop

    end % ends the restart loop
  else 
    if strcmp(matens,'dct')
      options=gagaOptions('tol',Tol,'maxiter',maxIter,'vecDistribution',vecDistribution,...
                      'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on');
    else
      options=gagaOptions('tol',Tol,'maxiter',maxIter,'vecDistribution',vecDistribution,'matrixEnsemble',matrixEnsemble,...
                      'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on');
    end 



    for ii=1:length(kmn_list{me})
  
      k=kmn_list{me}(ii,1);
      m=kmn_list{me}(ii,2);
      n=kmn_list{me}(ii,3);

      for p=1:tests_per_k
        if strcmp(matens,'smv')
          [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, nz, options);
        else
          [en, ct, it, supset, rate, xout] = gaga_cs(alg, matens, k,m,n, options);
        end
      end % ends the tests_per_k loop
  
      if strcmp(matens,'smv')
        display(sprintf('%s %s with %d nonzeros, n=%d, m=%d, and k=%d, completed after %f seconds.',alg,matens,nz,n,m,k,toc(ttt)))
      else
        display(sprintf('%s %s, n=%d, m=%d, and k=%d, completed after %f seconds.',alg,matens,n,m,k,toc(ttt)))
      end

    end % ends the kmn_list{me} loop

  end % ends if strcmp(alg,'CGIHT')
end % ends noise_list loop


  end %ends alg_list loop

end % ends vecDistr_list loop
end % ends nz_list loop for the nozeros_list with smv
end % ends matens_list loop

