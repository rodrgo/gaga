function process_smv_data(alg_list, vecDistribution)
%alg_list must be a cell of names

%alg_list=cell(1,1);
%alg_list{1}='ER_l0_M_smv';

%Modified to handle 'status'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fnames=ls;
fname_start='gpu_data_smv';
%look for files with the above starting string
ind=[];
for j=1:length(fnames)-length(fname_start)
  if strcmp(fname_start,fnames(j:j-1+length(fname_start)))
    ind=[ind j+length(fname_start)];
  end
end

%extract minimum and maximum dates strings
startdate=fnames(ind(1):ind(1)+7);
enddate=startdate;
for j=2:length(ind)
  tmp=fnames(ind(j):ind(j)+7);
  if str2num(tmp)<str2num(startdate)
    startdate=tmp;
  elseif str2num(tmp)>str2num(enddate)
    enddate=tmp;
  end
end

date_string=startdate;

results=[];
results_long=[];
results_full=[];
while (str2num(date_string)<=str2num(enddate))
  fname=[fname_start date_string '.txt'];

  fid = fopen(fname);
  if fid~=-1
    results_long = textscan(fid,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');
    fclose(fid);

    if strcmp(date_string,startdate)==1
      results_full=results_long;
    else
      for j=1:size(results_long,2)
        results_full{j}=[results_full{j}; results_long{j}];
      end
    end
    clear results_long
  end
  
  date_string=num2str(str2num(date_string)+1);
end

  
%the data stored in results is:
%1, algorithm
%4, k
%7, m
%10, n
%13, vecDistribution
%15, nonzeros_per_column
%17, matrixEnsemble
%19, error in l1 norm
%21, error in l2 norm
%23, error in l_infinity norm
%25, time for the algorithm
%27, time per iteration
%29, total time
%31, iterations
%33, convergence rate
%35, number of true positive 
%37, number of false positive
%39, number of true negative
%41, number of flase negative
%43, random number generator seed
%45, band_percentage 

columns_used=[4 7 10 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45];
columns='values in the columns of results are: k ,m, n,vecDistribution,nonzeros per column, matrix ensemble, l_one error,l_two error,l_infinity error, time for the algorithm, time per iteration,time total, iterations, convergence rate, number of true positive, number of false positives,number of true negatives, number of false negatives, and random number genertor seed band_percentage';

for j=1:length(alg_list)
  ind=find(strcmp(results_full{1},alg_list{j}));
  results=zeros(length(ind),length(columns_used));
  for pp=1:length(columns_used)
    results(:,pp)=results_full{columns_used(pp)}(ind);
  end
  
  fname_save=['results_' alg_list{j} '.mat'];
  save(fname_save,'results','columns');

end

% end function
end
