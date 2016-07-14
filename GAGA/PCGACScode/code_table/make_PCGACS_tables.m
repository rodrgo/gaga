%this script generates gpu data for any matrix ensemble, vector distribution,
%noise level, algorithm, etc. 

% 1. add the path to GAGA when using this script individually
%addpath /home/blanchaj/CurrentGAGA/GAGA/gaga

% 2. set the kmn_list for the tables directly
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



% 3 set the matrix ensembles you wish to test.  First choose from 'dct','smv','gen' and then 
% set the distribution of values in 'gen' or 'smv'
matens_list = cell(3,1);
matens_list{1} = 'dct';
matens_list{2} = 'smv';
matens_list{3} = 'gen';


% 4.  If using the 'smv' matrix ensembel, set all desired choices for the number of nonzeros per column.
nonzeros_list=[7]; % [4 7 13]

% 5.  Set the algorithms you want to test.

alg_list=cell(3,1);
alg_list{1}='NIHT';
alg_list{2}='HTP';
alg_list{3}='CSMPSP';


% 5. set all vector distributions with three total options: 'binary' (default), 'guassian', and 'uniform'.
vecDistr_list=cell(1,1);
vecDistr_list{1}='binary';
%vecDistr_list{2}='gaussian';
%vecDistr_list{3}='uniform';

% 6.  Set all noise levels you wish to test where the noise levels are a scaling factor for ||y||=||Ax||.
noise_list = [0 0.1]



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The following code creates all tables for the options you have chosen in 1.-11. %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttt=tic;

for vd=1:length(vecDistr_list)
  vecDistribution = vecDistr_list{vd};
  if strcmp(vecDistribution,'binary')
    vecDistr=1;
  elseif strcmp(vecDistribution,'gaussian')
    vecDistr=2;
  elseif strcmp(vecDistribution,'uniform')
    vecDistr=0;
  else
    error('Incorrect setting for vector distribution')
  end


 for me=1:length(matens_list)
   matens = matens_list{me};
   if strcmp(matens,'smv')
     nz_list = nonzeros_list;
   else
     nz_list = nonzeros_list(1);    % this is not needed for dct or gen, so the nz_list loop will be length 1.
   end


   for zz=1:length(nz_list);
     nz = nz_list(zz);

     for nl=1:length(noise_list)
       noise_level = noise_list(nl);

       make_table_PCGACS(alg_list,matens,nz,noise_level,vecDistr,kmn_list{me})

     end % ends noise_list loop
   end % ends nz_list loop
 end % ends matens_list loop
end % ends vecDistr_list loop 
