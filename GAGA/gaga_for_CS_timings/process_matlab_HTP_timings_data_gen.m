fnames=ls;
fname_start='matlab_timings_HTP_data_gen';
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


%startdate='20120328';
%enddate='20120329';

date_string=startdate;

alg='HTP_M_gen';

vecDistribution=1; %0 is for uniform [0,1], 
                   %1 is for random {-1,1},
                   %2 is for gaussian.

matrixEnsemble=1; %1 for gaussian,
                  %2 for uniform {-1,1}

alpha=0.0;
supp_flag=0; %(0 is for dynamic and 1 is for always binning)
		%(2 is for sort with dynamic and 3 is for always sort)

results_long=cell(0,46);
tmp=cell(1,46);
tic

fname_save=sprintf('results_HTP_gen_matlab_timing_supp_flag_%d.mat',supp_flag);
fid=fopen(fname_save);

% do nothing if the data has already been processed
if fid~=-1
  fclose(fid);
  load(fname_save);
else
  %the below processes the data if it has not already been processed  

  while (str2num(date_string)<=str2num(enddate))
  
    fname=['matlab_timings_HTP_data_gen' date_string '.txt'];
    fid = fopen(fname);

    if fid~=-1
      %data is read into an array of cells with 46 columns.  what is
      %stored in each column is listed below.

      while(1)
        p=0;
        results_short=cell(0,46);
        while (p<1000) 
          tmp1 = textscan(fid,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f',1);
          if (prod(size(tmp1{1}))==0)  break; end
          tmp2 = textscan(fid,'%s %f %s %f %s %f %s %s %f %s %s %f',tmp1{1,27});
          for ll=1:45
            tmp{ll}=tmp1{ll};
          end
          tmp{46}=[tmp2{2} tmp2{4} tmp2{6} tmp2{9} tmp2{12}];
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

  
  
  
  num_tests=size(results_long,1);

  %extract the data for NIHT_S_gen with the specified
  %vecDistribution, alpha, and supp_flag
  ind=[];
  for j=1:num_tests
    if (strcmp(results_long{j,1},alg) & results_long{j,13}==vecDistribution & abs(results_long{j,41}-alpha)<0.001 & results_long{j,43}==supp_flag & results_long{j,45}==matrixEnsemble)
      ind = [ind; j];
    end
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
%21, time without createProblem
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
%               possible, if equal to 1 then does full binning at
%               each setp).
%45, matrixEnsemble (1 gaussian and 2 uniform {-1,1})
%46, a 5 by iterations array with the first column being 
%    the iteration number, the second the time for that iteration (total
%    time) and the third column the time to find the support set
%    for that iteration, fourth column is the number of CG steps in
%    that iteration and the fifth column is how many CG iteration
%    were used in that iteration.


%for this file we use items [4, 7, 10, 13, 19, 25, 27, 29, 31, 33, 44]
%as this is eleven entries, we strip results into an 11 column array


  results_long=results_long(ind,:);
  toc

  
  k=[results_long{:,4}]';
  m=[results_long{:,7}]';
  n=[results_long{:,10}]';
  
  iter_max=0;
  for j=1:length(k)
    iter_max=max(iter_max,size(results_long{j,46},1));
  end

  
  results=cell(0);
  results_ind=[19 21 25];
  results_content=sprintf('columns are: infty_error, total time without problem creation, total time with problem creation');

  time_per_iteration_kmn_all=cell(0);
  time_supp_set_kmn_all=cell(0);
  time_per_cg_iteration_kmn_all=cell(0);
  total_cg_steps_kmn_all=cell(0);

  %look for independent k_m_n so that tests for the same k_m_n can be combined.
  kmn=[k m n]; %kmn_list=intersect(kmn,kmn,'rows');
  %for each m,n pair there may be a few very similar values of k
  %which should be combined.  tests were done on a spacing of 0.02
  %in rho so cluster those that are closest.
  rho_grid=0.02:0.02:0.98;
  
  mn=[m n]; mn_list=intersect(mn,mn,'rows');
  kmn_list=[];
  for j=1:length(mn_list);
    k_width=mn_list(j,1)*rho_grid(1)/2; 
    k_grid=round(mn_list(j,1)*rho_grid);
    k_grid=intersect(k_grid,k_grid);
    for ll=1:length(k_grid)
      k_trial=k_grid(ll);
      ind_m=find(m==mn_list(j,1));
      ind_n=find(n==mn_list(j,2));
      ind_k=find( (k>=k_trial-k_width) & (k<k_trial+k_width) );
      ind=intersect(ind_m,ind_n);
      ind=intersect(ind,ind_k);
      if min(size(ind))>0
        kmn_list=[kmn_list; [k_trial mn_list(j,:)]];
        
        %for each different kmn triple put all times into a long
        %vector
        results_tmp=[];
        time_tmp_per=[];
        time_tmp_supp=[];
        time_cg=[];
        cg_steps=[];
        
        for lll=1:length(ind)
          
          tmp=[];
          for zz=1:length(results_ind);
            tmp=[tmp results_long{ind(lll),results_ind(zz)}];
          end
          results_tmp=[results_tmp; tmp];
          
          time_tmp_per=[time_tmp_per; results_long{ind(lll),46}(:,2)-results_long{ind(lll),46}(:,3)-results_long{ind(lll),46}(:,5)];
          time_tmp_supp=[time_tmp_supp; results_long{ind(lll),46}(:,3)];
          cg_steps=[cg_steps; sum(results_long{ind(lll),46}(:,4))];
          ind_tmp=find(results_long{ind(lll),46}(:,4)>0);
          time_cg=[time_cg; results_long{ind(lll),46}(ind_tmp,5)./results_long{ind(lll),46}(ind_tmp,4)];
        
        end
        
        
        a=cell(1);
        a{1}=results_tmp;
        results=[results a];
        a{1}=sort(time_tmp_per);
        time_per_iteration_kmn_all=[time_per_iteration_kmn_all a];
        a{1}=sort(time_tmp_supp);
        time_supp_set_kmn_all=[time_supp_set_kmn_all a];        
        a{1}=sort(time_cg);
        time_per_cg_iteration_kmn_all=[time_per_cg_iteration_kmn_all a];
        a{1}=sort(cg_steps);
        total_cg_steps_kmn_all=[total_cg_steps_kmn_all a];
        
      end
    end
  end
  
  if prod(size(kmn_list))>0
    [kmn_list,ind]=sortrows(kmn_list,[3 2 1]);
    results=results(ind);
    time_per_iteration_kmn_all=time_per_iteration_kmn_all(ind);
    time_supp_set_kmn_all=time_supp_set_kmn_all(ind);
    time_per_cg_iteration_kmn_all=time_per_cg_iteration_kmn_all(ind);
    total_cg_steps_kmn_all=total_cg_steps_kmn_all(ind);
  end
  
  
  save(fname_save,'kmn_list','time_per_iteration_kmn_all','time_supp_set_kmn_all','time_per_cg_iteration_kmn_all','total_cg_steps_kmn_all','results','results_content');

end





