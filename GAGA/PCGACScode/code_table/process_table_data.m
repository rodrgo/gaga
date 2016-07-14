
%turn on (1) or off (0) ensembles to be processed.
dct = 1;
smv = 1;
gen = 1;

% to process the data, set process = 1, but to only recompute the logit data, set process = 0
process = 1;

% to use a specific range of dates, set specific_dates=1, otherwise specific_dates=0 will find all files in the folder
specific_dates = 0;

% to add logit data set logit_on=1
logit_on = 0;  % no logit data is required for the table data

processTic=tic;
% set the noise values
noise_list = [0 0.1]; %[0 0.1];

% set the vector distributions
vecDistr_list = [1]; % 2 0];

% set the start and end dates for the data files
startdate = '20140127';
enddate = '20140127';

% optional way to specify dates based on ensemble
startdate_dct = startdate;
enddate_dct = enddate;
startdate_smv = startdate;
enddate_smv = enddate;
startdate_gen = startdate;
enddate_gen = enddate;


% initialization of ens_list, no need to change this.
ens_list=cell(0,1);


% THe algorithms for which data with noise or alternative distributions should be processed. 
% A list not depenedent on matrix ensemble; a list for each ensemble is generated automatically.

alg_list=cell(3,1);
alg_list{1}='NIHT';
alg_list{2}='HTP';
alg_list{3}='CSMPSP';


 

if dct == 1

  alg_list_dct=cell(length(alg_list),1);
  for i=1:length(alg_list)
    alg_list_dct{i}=[alg_list{i} '_S_dct'];
  end
  ens_list = [ens_list 'dct'];

  if process == 1
    if specific_dates == 1
      process_dct_data_noise(alg_list_dct, noise_list, vecDistr_list, startdate_dct, enddate_dct);
    else
      process_dct_data_noise(alg_list_dct, noise_list, vecDistr_list);
    end
  end

end % end if dct==1

if smv == 1

  alg_list_smv=cell(length(alg_list),1);
  for i=1:length(alg_list)
    alg_list_smv{i}=[alg_list{i} '_S_smv'];
  end
  ens_list = [ens_list 'smv'];

  if process == 1
    if specific_dates == 1
      process_smv_data_noise(alg_list_smv, noise_list, vecDistr_list, startdate_smv, enddate_smv);
    else
      process_smv_data_noise(alg_list_smv, noise_list, vecDistr_list);
    end
  end

end % end if smv==1

if gen == 1

  alg_list_gen=cell(length(alg_list),1);
  for i=1:length(alg_list)
    alg_list_gen{i}=[alg_list{i} '_S_gen'];
  end
  ens_list = [ens_list 'gen'];

  if process == 1
    if specific_dates == 1
      process_gen_data_noise(alg_list_gen, noise_list, vecDistr_list, startdate_gen, enddate_gen);
    else
      process_gen_data_noise(alg_list_gen, noise_list, vecDistr_list);
    end
  end

end  % end if gen==1

process_time = toc(processTic);
fprintf('Processing Completed after %f seconds.\n Begin add_logit_data.',process_time);

% Now we add logit data to all files we just created.
% This portion of the process can take some time.
if logit_on
  add_logit_data_all_noise(alg_list, ens_list, noise_list, vecDistr_list)
end
