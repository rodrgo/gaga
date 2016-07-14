function add_logit_data_all(alg_list, ens_list, noise_list, vecDistr_list)
% add_logit_data_all_noise is the version of 

addpath ../

if nargin<4
  vecDistr_list = [1 2 0];
  warning('No vector distribution list specified by user; default [1 2 0] used.')
  if nargin<3
    noise_list = [0 0.1];
    warning('No noise list specified by user; default [0 0.1] used.')
    if nargin<2
      ens_list=cell(1,1);
      ens_list{1}='gen';
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
  end
end 


ttt=tic;

for i=1:length(alg_list)
  for j=1:length(ens_list)
    for pp=1:length(noise_list)
      for qq=1:length(vecDistr_list)
        add_logit_data_noise(alg_list{i},ens_list{j},noise_list(pp),vecDistr_list(qq));
        [i length(alg_list) j length(ens_list) pp length(noise_list) qq length(vecDistr_list)]
        thisone = toc
        totaltime = toc(ttt)
      end
    end
  end
end


