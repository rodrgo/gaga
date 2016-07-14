function process_dct_data_noise(alg_list, noise_list, vecDistr_list, startdate, enddate)
% process_dct_data_noise(alg_list, noise_list, vecDistr_list, startdate, enddate)
% Inputs:  (all inputs are optional with default values included in the function)
%  alg_list is a cell containing the algorithms for which the data should be processed; entries must end with '_S_dct'
%  noise_list is a vector of noise level values
%  vecDistr_list is a vector of integers identifying the vector distributions 1 = binary, 0 = uniform, 2 = Guassian
%  startdate (optional) is the beginning of a specified date range
%  enddate (optional) is the end of a specified date range
% process_dct_data_noise ends the name of each .mat file with the vector distribution: '_vecDistrX' where X is from vecDistr_list

no_zero_string = 1;  % older versions of GAGA should have this set to 0 for noise tests

specific_dates = 1;  
if ( nargin==5 ) && ( str2num(startdate)>str2num(enddate) )
  error('The end date is earlier than the start date.');
end

if nargin<5
 if nargin==4
  enddate = datestr(now,'yyyymmdd');  % if startdate is provided, but no enddate, the enddate is set to the current date
 elseif nargin<4
  specific_dates = 0;   % no date range was provided; all data files in the folder will be used
  if nargin<3
    vecDistr_list = [1 2 0];
    warning('No vector distribution list specified by user; default [1 2 0] used.')
    if nargin<2
      noise_list = [0 0.1];
      warning('No noise list specified by user; default [0 0.1] used.')
      if nargin<1
        alg_list=cell(4,1);
	alg_list{1}='CGIHT_S_dct';
	alg_list{2}='NIHT_S_dct';
	alg_list{3}='HTP_S_dct';
	alg_list{4}='CSMPSP_S_dct';
	warning('No algorithm list specified by user; default {CGIHT, NIHT, HTP, CSMPSP} used.')
      end
    end
  end
 end
end


for qq=1:length(noise_list)
    noise_level = noise_list(qq);
    noise_string = ['_noise' num2str(noise_level)];
    % The noise string must be 5 characters x.xxx so we append zeros as
    % necessary.
    switch length(num2str(noise_level))
        case 1
            noise_string = [noise_string '.' num2str(0) num2str(0) num2str(0)];
        case 2
            error('The noise_levels must be either an integer or have between one and three decimal places.')
        case 3
            noise_string = [noise_string num2str(0) num2str(0)];
        case 4
            noise_string = [noise_string num2str(0)];
        otherwise 
            error('The noise_levels must be either an integer or have between one and three decimal places.')
    end

    if (noise_level == 0) && no_zero_string
      noise_string = '';
    end

  if specific_dates
    filenames = get_data_file_names('dct',noise_level,no_zero_string,startdate,enddate);
  else
    filenames = get_data_file_names('dct',noise_level,no_zero_string);
  end

  results=[];
  results_long=[];
  results_full=[];
  for jj=1:length(filenames)

    fname=filenames{jj};

    fid = fopen(fname);
    if fid~=-1
      results_long = textscan(fid,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');
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
  end  %ends loop through filenames
  
%the data stored in results is:
%1, algorithm
%4, k
%7, m
%10, n
%13, vecDistribution
%15, error in l1 norm
%17, error in l2 norm
%19, error in l_infinity norm
%21, time for the algorithm
%23, time per iteration
%25, total time
%27, iterations
%29, convergence rate
%31, number of true positive 
%33, number of false positive
%35, number of true negative
%37, number of flase negative
%39, random number generator seed

  columns_used=[4 7 10 13 15 17 19 21 23 25 27 29 31 33 35 37 39];
  columns='values in the columns of results are: k ,m, n,vecDistribution,l_one error,l_two error,l_infinity error, time for the algorithm, time per iteration,time total, iterations, convergence rate, number of true positive, number of false positives,number of true negatives, number of false negatives, and random number genertor seed';

  for ww=1:length(vecDistr_list)
    vecDistribution = vecDistr_list(ww);
    vec_str = ['_vecDistr' num2str(vecDistribution)];

    for j=1:length(alg_list)
      ind=find( (strcmp(results_full{1},alg_list{j})) .* (results_full{13}==vecDistribution) );
      results=zeros(length(ind),length(columns_used));
      for pp=1:length(columns_used)
        results(:,pp)=results_full{columns_used(pp)}(ind);
      end
  
      fname_save=['results_' alg_list{j} vec_str noise_string '.mat'];
      save(fname_save,'results','columns');
    end % ends for loop to save mat file.

  end % ends for loop for vecDistr_list

end % ends for loop on noise_list.

