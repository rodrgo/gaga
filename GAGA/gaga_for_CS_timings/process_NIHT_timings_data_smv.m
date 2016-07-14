fnames=ls;
fname_start='gpu_timings_data_smv';
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

%startdate='20120325';
%enddate='20120327';

date_string=startdate;

alg='NIHT_S_smv';

vecDistribution=1; %0 is for uniform [0,1], 
                   %1 is for random {-1,1},
                   %2 is for gaussian.

alpha=0.25;
supp_flag=0; %(0 is for dynamic and 1 is for always binning)
		%(2 is for sort with dynamic and 3 is for always sort)

%nonZerosPerColumn=4;

matrixEnsemble=2; %1 is for all nonzeros equal to 1
                  %2 is for nonzeros random {-1,1}.

results_long=cell(0,48);
tmp=cell(1,48);
tic

%check if the desired .mat files exist
data_ready=1;
for supp_flag=0:0
  fname_save=sprintf('results_NIHT_smv_timing_supp_flag_%d.mat',supp_flag);
  fid=fopen(fname_save);
  if fid~=-1 fclose(fid); end %if the file exists then close it
  if fid==-1
    data_ready=0;
    break;
  end
end


if (data_ready==0)
  %the below processes the data if it has not already been processed  

  while (str2num(date_string)<=str2num(enddate))

  fname=['gpu_timings_data_smv' date_string '.txt'];
  fid = fopen(fname);
  if fid~=-1
    display(date_string)
    %data is read into an array of cells with 46 columns.  what is
    %stored in each column is listed below.

    while(1)
      p=0;
      results_short=cell(0,48);
      while (p<1000)
        tmp1 = textscan(fid,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f',1);
        if (prod(size(tmp1{1}))==0)  break; end
        tmp2 = textscan(fid,'%s %f %s %f %s %f',tmp1{1,27});
        for ll=1:47
          tmp{ll}=tmp1{ll};
        end
        tmp{48}=[tmp2{2} tmp2{4} tmp2{6}];
        results_short = [results_short; tmp];
        p=p+1;
      end
      results_long = [results_long; results_short];
      if (prod(size(tmp1{1}))==0)  break; end
      toc
    end
    fclose(fid);
  end

    
    date_string=num2str(str2num(date_string)+1);
  end



%the data stored in results is:
%1, algorithm
%4, k
%7, m
%10, n
%13, vecDistribution
%15, error in l1 norm
%17, error in l2 norm
%19, error in l_infinity norm
%21, time???
%23, time per iteration
%25, total time
%27, iterations
%29, convergence rate
%31, number of true positive 
%33, number of false positive
%35, number of true negative
%37, number of flase negative
%39, random number generator seed
%41, alpha \in [0,1), with 0 using the full number of bins from the
%    start and as 1 is approached the number of bins goes to 1
%43, supp_flag (if equal to 0 then avoids counting bins if
%               possible, if equal to 1 then does full binning at each setp).
%45, p = Nonzeros per column.
%47, matrixEnsemble, 1 for all nonzeros ones, 2 for nonzeros random {-1,1}
%48, a 3 by iterations array with the first column being 
%    the iteration number, the second the time for that iteration (total
%    time) and the third column the time to find the support set
%    for that iteration.


%for this file we use items [4, 7, 10, 13, 19, 25, 27, 29, 31, 33, 45, 47, 48]
%as this is eleven entries, we trip results into a 13 column array

  results_full=results_long;
  num_tests=size(results_full,1);


  for supp_flag=0:0

    %extract the data for NIHT_S_smv with the specified
    %vecDistribution, alpha, supp_flag, and matrixEnsemble
    ind=[];
    for j=1:num_tests
      if (strcmp(results_full{j,1},alg) ...
          & results_full{j,13}==vecDistribution ...
          & abs(results_full{j,41}-alpha)<0.001 ...
          & results_full{j,43}==supp_flag ...
          & results_full{j,47}==matrixEnsemble)
        ind = [ind; j];
      end
    end
  
    results_long=results_full(ind,:);
    toc

  
    k=[results_long{:,4}]';
    m=[results_long{:,7}]';
    n=[results_long{:,10}]';
    p=[results_long{:,45}]'; %number of nonzeros per column

    iter_max=0;
    for j=1:length(results_long)
      iter_max=max(iter_max,size(results_long{j,48},1));
    end
    
    
    results=cell(0);
    results_ind=[19 21 25];
    results_content=sprintf('columns are: infty_error, total time without problem creation, total time with problem creation');

    time_per_iteration_kmnp_all=cell(0);
    time_supp_set_kmnp_all=cell(0);
  
  
    %look for independent k_m_n_p so that tests for the same k_m_n_p can be averaged.
    kmnp=[k m n p]; %kmnp_list=intersect(kmnp,kmnp,'rows');
    %for each m,n,p tripple there may be a few very similar values of k
    %which should be combined.  tests were done on a spacing of 0.02
    %in rho so cluster those that are closest.
    rho_grid=0.02:0.02:0.98;
  
    mnp=[m n p]; mnp_list=intersect(mnp,mnp,'rows');
    kmnp_list=[];
    for j=1:length(mnp_list);
      k_width=mnp_list(j,1)*rho_grid(1)/2; 
      k_grid=round(mnp_list(j,1)*rho_grid);
      k_grid=intersect(k_grid,k_grid);
      for ll=1:length(k_grid)
        k_trial=k_grid(ll);
        ind_m=find(m==mnp_list(j,1));
        ind_n=find(n==mnp_list(j,2));
        ind_p=find(p==mnp_list(j,3));
        ind_k=find( (k>=k_trial-k_width) & (k<k_trial+k_width) );
        ind=intersect(ind_m,ind_n);
        ind=intersect(ind,ind_p);
        ind=intersect(ind,ind_k);
        if min(size(ind))>0
          kmnp_list=[kmnp_list; [k_trial mnp_list(j,:)]];
      
          
          %for each different kmnp put all times into a long
          %vector
          results_tmp=[];
          time_tmp_per=[];
          time_tmp_supp=[];
          for lll=1:length(ind)
            
            tmp=[];
          for zz=1:length(results_ind);
            tmp=[tmp results_long{ind(lll),results_ind(zz)}];
          end
          results_tmp=[results_tmp; tmp];
          
            time_tmp_per=[time_tmp_per; results_long{ind(lll),48}(:,2)-results_long{ind(lll),48}(:,3)];
            time_tmp_supp=[time_tmp_supp; results_long{ind(lll),48}(:,3)];
          
          end
        
          a=cell(1);
          a{1}=results_tmp;
          results=[results a];
          a{1}=sort(time_tmp_per);
          time_per_iteration_kmnp_all=[time_per_iteration_kmnp_all a];
          a{1}=sort(time_tmp_supp);
          time_supp_set_kmnp_all=[time_supp_set_kmnp_all a];        
        
        end      
      end
    end
  
    if prod(kmnp_list)>0
      [kmnp_list,ind]=sortrows(kmnp_list,[3 2 1 4]);
      results=results(ind);
      time_per_iteration_kmnp_all=time_per_iteration_kmnp_all(ind);
      time_supp_set_kmnp_all=time_supp_set_kmnp_all(ind);
    end
    
    fname_save=sprintf('results_NIHT_smv_timing_supp_flag_%d.mat',supp_flag);
    save(fname_save,'kmnp_list','time_per_iteration_kmnp_all','time_supp_set_kmnp_all','results','results_content');
  end

  clear results_long results_short ind iter

end
  

