function add_logit_data_all(alg_list, ens_list)
% add_logit_data_all determines the logistic regression for the phase transition
% for noise free data from problem class (Mat,B)
% For data with noise or for other vector distrubtions with or without noise,
% use the function add_logit_data_all_noise.m

addpath ../

if nargin<2
  ens_list=cell(1,1);
  ens_list{1}='dct';
  warning('No ensemble list specified by user; default {dct} used.')
  if nargin<1
    alg_list=cell(4,1);
    alg_list{1}='ThresholdCG';
    alg_list{2}='NIHT';
    alg_list{3}='HTP';
    alg_list{4}='CSMPSP';
    warning('No algorithm list specified by user; default {ThresholdCG, NIHT, HTP, CSMPSP} used.')
  end
end
 


ttt=tic;

for i=1:length(alg_list)
  for j=1:length(ens_list)
    add_logit_data(alg_list{i},ens_list{j});
    [i length(alg_list) j length(ens_list)]
    thisone = toc
    totaltime = toc(ttt)
  end
end


