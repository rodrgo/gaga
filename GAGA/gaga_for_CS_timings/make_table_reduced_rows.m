function [table_reduced table_reduced_choice] = make_table_reduced_rows(alg,processor,matrixEnsemble,supp_flag,tableType,sds_to_add,nonZeros_select)

% alg will be either: NIHT or HTP
% processor will be either: gpu or matlab
% matrixEnsemble will be either: dct, smv, or gen
% n_select is the number of columns
% supp_flag is: 0 for dynamic binning, 1 is for always binning,
%               2 for sort with dynamics, and 3 is for always sorting
% 
% nonZeros_select is only for matrixEnsemble smv and is the number of
% nonzeros per column
% tableType selects what kind of table is to be generated, values are:
% 1 is for time_per_iteration
% 2 is for time_support_set
% 3 is for time_per_cg_iteration
% 4 is for total_cg_iterations
% 5 is for the problem generation
% sds_to_add is the number of standard deviations to add to the
% data: in particular, the data used to generate the table is that
% specified by tableType and with sds_to_add of its standard
% deviations added.


% if the tableType is two digits then the first digit designates
% the type as above and the second digit is how many standard
% deviations should be added to the data.
% if the tableType is three digits then the first digit designates
% the type as above, the second digit is how many standard
% deviations should be added to the data, and the third digit
% should be zero.
% (for example 12 is time_per_iteration + 2 standard deviations and 
%  310 is time_per_cg_iterations - 1 standard deviation.)

if (tableType==3 | tableType==4) & strcmp(alg,'NIHT')
  error('NIHT requires tableType to be 1, 2 or 5');
elseif (tableType>5) & strcmp(alg,'HTP')
  error('HTP requires tableType to be 1, 2, 3, 4, or 5');
end



% the other variants are always one choice in the data considered here
vecDistribution=1; %0 is for uniform [0,1], 
                   %1 is for random {-1,1},
                   %2 is for gaussian.


if strcmp(processor,'gpu')
  fname_save=sprintf('results_%s_%s_timing_supp_flag_%d',alg,matrixEnsemble,supp_flag);
elseif strcmp(processor,'matlab')
  fname_save=sprintf('results_%s_%s_matlab_timing_supp_flag_%d',alg,matrixEnsemble,supp_flag);
else
  error('invalid processor type, should be gpu or matlab');
end
load(fname_save)


%USE NARGIN FOR NONZEROS
% take the data with the specified n_select, and for m>100 (where m
% is the number of rows), and if nonZeros is specified then select
% only that portion of the data for smv.
if nargin==7
  ind=find(kmnp_list(:,4)==nonZeros_select);
  kmn_list=kmnp_list(ind,1:3);
  results_used=results(ind);
  time_per_iteration_kmn_all=time_per_iteration_kmnp_all(ind);
  time_supp_set_kmn_all=time_supp_set_kmnp_all(ind);
  if strcmp(alg,'HTP')
    time_per_cg_iteration_kmn_all=time_per_cg_iteration_kmnp_all(ind);
    total_cg_steps_kmn_all=total_cg_steps_kmnp_all(ind);
  end
else
  results_used=results;
end

%need to check that each kmn_list triple has enough data to make a
%sensible table.  

if strcmp(alg,'HTP') & (tableType==3 | tableType==4)
  num_of_data_per_kmn=zeros(size(kmn_list,1),1);
  for jj=1:length(num_of_data_per_kmn)
    num_of_data_per_kmn(jj)=length(time_per_cg_iteration_kmn_all{jj});
  end
  min_num_cg=5;
  ind=find(num_of_data_per_kmn>=min_num_cg);

  kmn_list=kmn_list(ind,1:3);
  time_per_iteration_kmn_all=time_per_iteration_kmn_all(ind);
  time_supp_set_kmn_all=time_supp_set_kmn_all(ind);

  time_per_cg_iteration_kmn_all=time_per_cg_iteration_kmn_all(ind);
  total_cg_steps_kmn_all=total_cg_steps_kmn_all(ind);
end




%************************************************
% The data is now ready, and tables can be formed
%


%compute the average and standard deviation of the measured quantities
data_for_table=zeros(length(kmn_list),1);
data_for_table_sd=zeros(length(kmn_list),1);

if tableType==1
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(time_per_iteration_kmn_all{zz})/length(time_per_iteration_kmn_all{zz});
    data_for_table_sd(zz)=norm(time_per_iteration_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(time_per_iteration_kmn_all{zz}));
  end
elseif tableType==2
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(time_supp_set_kmn_all{zz})/length(time_supp_set_kmn_all{zz});
    data_for_table_sd(zz)=norm(time_supp_set_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(time_supp_set_kmn_all{zz}));
  end
elseif tableType==3
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(time_per_cg_iteration_kmn_all{zz})/length(time_per_cg_iteration_kmn_all{zz});
    data_for_table_sd(zz)=norm(time_per_cg_iteration_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(time_per_cg_iteration_kmn_all{zz}));
  end
elseif tableType==4
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(total_cg_steps_kmn_all{zz})/length(total_cg_steps_kmn_all{zz});
    data_for_table_sd(zz)=norm(total_cg_steps_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(total_cg_steps_kmn_all{zz}));
  end
elseif tableType==5
  for zz=1:length(kmn_list)
    prob_gen_times=results_used{zz}(:,3)-results_used{zz}(:,2);
    data_for_table(zz)=sum(prob_gen_times)/length(prob_gen_times);
    data_for_table_sd(zz)=norm(prob_gen_times-data_for_table(zz),2)/sqrt(length(prob_gen_times));
  end
end
data_for_table=data_for_table + sds_to_add*data_for_table_sd;


% compute the fits for each value of n separately
n_list=intersect(kmn_list(:,3),kmn_list(:,3));



small_m=100;

table_reduced=cell(length(n_list),1);
table_reduced_choice=zeros(length(n_list),1);

for j=1:length(n_list)
  n_select=n_list(j);

  ind=find( (kmn_list(:,3)==n_select) & (kmn_list(:,2)>small_m)); 

  d=kmn_list(ind,2)/n_select;
  r=kmn_list(ind,1)./kmn_list(ind,2);
  m_list=intersect(kmn_list(ind,2),kmn_list(ind,2));


  % fit time_per_iteration with c_1+c_2\delta+c_3\rho as well as
  % simpler options (constant, delta, rho, and delta + rho)

  A1=[ones(size(d))];
  A2=[ones(size(d)) d];
  A3=[ones(size(d)) r];
  A4=[ones(size(d)) d r];

  coef1=pinv(A1)*data_for_table(ind);
  coef2=pinv(A2)*data_for_table(ind);
  coef3=pinv(A3)*data_for_table(ind);
  coef4=pinv(A4)*data_for_table(ind);

  err1=A1*coef1-data_for_table(ind);
  err2=A2*coef2-data_for_table(ind);
  err3=A3*coef3-data_for_table(ind);
  err4=A4*coef4-data_for_table(ind);
  
  
  %possibly remove one outlier
  err1=sort(abs(err1),'descend');
  if err1(1)>10*err1(2)
    err1(1)=err1(2);
  end
  err2=sort(abs(err2),'descend');
  if err2(1)>10*err2(2)
    err2(1)=err2(2);
  end
  err3=sort(abs(err3),'descend');
  if err3(1)>10*err3(2)
    err3(1)=err3(2);
  end
  err4=sort(abs(err4),'descend');
  if err4(1)>10*err4(2)
    err4(1)=err4(2);
  end
  
  
  %choose which regression to use for the reduced tables, use the
  %amount of the l2 error increase to automatically select a
  %regression form that is both accurate and simple.
  reduced_flag=1;
  e1=norm(err1,2);  e2=norm(err2,2);  e3=norm(err3,2);  e4=norm(err4,2);
  if (min(e2,e3) < (0.90*e1))
    if (e2 < e3) %&& (abs(coef2(2)/coef2(1))>0.1) %linear part is at
                                               %least 10% of fit
      reduced_flag=2;
    else %if (abs(coef3(2)/coef3(1))>0.1)
      reduced_flag=3;
    end
    if e4 < (0.7*min(e2,e3))
      reduced_flag=4;
    end
  end
  table_reduced_choice(j)=reduced_flag;
  

  %now print the selected fit to the reduced table
  if reduced_flag==1
    table_reduced{j}=sprintf('$2^{%d}$ & %5.3g & - & - & %5.3g\\\\\\\\',log2(n_select),coef1(1),norm(err1,inf));
  elseif reduced_flag==2
    table_reduced{j}=sprintf('$2^{%d}$ & %5.3g & %5.3g & - & %5.3g\\\\\\\\',log2(n_select),coef2(1),coef2(2),norm(err2,inf));
  elseif reduced_flag==3
    table_reduced{j}=sprintf('$2^{%d}$ & %5.3g & - & %5.3g & %5.3g\\\\\\\\',log2(n_select),coef3(1),coef3(2),norm(err3,inf));
  elseif reduced_flag==4
    table_reduced{j}=sprintf('$2^{%d}$ & %5.3g & %5.3g & %5.3g & %5.3g\\\\\\\\',log2(n_select),coef4(1),coef4(2),coef4(3),norm(err4,inf));
  end
  
end



