function process_gen_data(alg_list, startdate, enddate)
% process_gen_data(alg_list, startdate, enddate)
% Inputs:  (all inputs are optional with default values included in the function)
%  alg_list is a cell containing the algorithms for which the data should be processed; entries must end with '_S_gen'
%  startdate (optional) is the beginning of a specified date range
%  enddate (optional) is the end of a specified date range
% This function process the data for all algorithms in the list for the matrix ensemble 'gen'
% and for sparse binary vectors with no noise, i.e. problem class (N, B).
% For noise or for other vector distributions, use process_gen_data_noise.


addpath ../shared/

specific_dates = 1;    % flag to determing if a date range was specified

if ( nargin==3 ) && ( str2num(startdate)>str2num(enddate) )
  error('The end date is earlier than the start date.');
end

if nargin<3
 if nargin==2
   enddate = datestr(now,'yyyymmdd');  % if startdate is provided, but no enddate, the enddate is set to the current date
 elseif nargin<2
   specific_dates = 0;   % no date range was provided; all data files in the folder will be used
   if nargin<1
     alg_list=cell(4,1);
     alg_list{1}='ThresholdCG_S_gen';
     alg_list{2}='NIHT_S_gen';
     alg_list{3}='HTP_S_gen';
     alg_list{4}='CSMPSP_S_gen';
     warning('No algorithm list specified by user; default {ThresholdCG, NIHT, HTP, CSMPSP} used.')
   end
 end
end

if specific_dates
  filenames = get_data_file_names('gen',0,1,startdate,enddate);
else
  filenames = get_data_file_names('gen');
end

results=[];
results_long=[];
results_full=[];
for jj=1:length(filenames)

  fname=filenames{jj};

  fid = fopen(fname);
  if fid~=-1
    results_long = textscan(fid,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');
    fclose(fid);

    if jj==1
      results_full=results_long;
    else
      for j=1:size(results_long,2)
        results_full{j}=[results_full{j}; results_long{j}];
      end
    end
    clear results_long
  end
end % ends loop through filenames

  
%the data stored in results is:
%1, algorithm
%4, k
%7, m
%10, n
%13, vecDistribution
%15, matrixEnsemble
%17, error in l1 norm
%19, error in l2 norm
%21, error in l_infinity norm
%23, time for the algorithm
%25, time per iteration
%27, total time
%29, iterations
%31, convergence rate
%33, number of true positive 
%35, number of false positive
%37, number of true negative
%39, number of flase negative
%41, random number generator seed


columns_used=[4 7 10 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41];
columns='values in the columns of results are: k ,m, n,vecDistribution,matrix ensemble, l_one error,l_two error,l_infinity error, time for the algorithm, time per iteration,time total, iterations, convergence rate, number of true positive, number of false positives,number of true negatives, number of false negatives, and random number genertor seed';

vecDistribution=1;

for j=1:length(alg_list)
  ind=find( (strcmp(results_full{1},alg_list{j})) .* (results_full{13}==vecDistribution) );
  results=zeros(length(ind),length(columns_used));
  for pp=1:length(columns_used)
    results(:,pp)=results_full{columns_used(pp)}(ind);
  end
  fname_save=['results_' alg_list{j} '.mat'];
  save(fname_save,'results','columns');

end % ends for loop to save mat file.


