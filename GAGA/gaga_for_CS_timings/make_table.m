function make_table(alg,processor,matrixEnsemble,supp_flag,tableType,sds_to_add,nonZeros_select)

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
  results=results(ind);
  time_per_iteration_kmn_all=time_per_iteration_kmnp_all(ind);
  time_supp_set_kmn_all=time_supp_set_kmnp_all(ind);
  if strcmp(alg,'HTP')
    time_per_cg_iteration_kmn_all=time_per_cg_iteration_kmnp_all(ind);
    total_cg_steps_kmn_all=total_cg_steps_kmnp_all(ind);
  end
end

%need to check that each kmn_list triple has enough data to make a
%sensible table.  

if strcmp(alg,'HTP')
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

fname_full='tables/';
fname_reduced='tables/';

%compute the average and standard deviation of the measured quantities
data_for_table=zeros(length(kmn_list),1);
data_for_table_sd=zeros(length(kmn_list),1);

if tableType==1
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(time_per_iteration_kmn_all{zz})/length(time_per_iteration_kmn_all{zz});
    data_for_table_sd(zz)=norm(time_per_iteration_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(time_per_iteration_kmn_all{zz}));
  end
  fname_full=[fname_full 'table_time_per_iteration_full_'];
  fname_reduced=[fname_reduced 'table_time_per_iteration_'];
elseif tableType==2
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(time_supp_set_kmn_all{zz})/length(time_supp_set_kmn_all{zz});
    data_for_table_sd(zz)=norm(time_supp_set_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(time_supp_set_kmn_all{zz}));
  end
  fname_full=[fname_full 'table_time_supp_set_full_'];
  fname_reduced=[fname_reduced 'table_time_supp_set_'];
elseif tableType==3
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(time_per_cg_iteration_kmn_all{zz})/length(time_per_cg_iteration_kmn_all{zz});
    data_for_table_sd(zz)=norm(time_per_cg_iteration_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(time_per_cg_iteration_kmn_all{zz}));
  end
  fname_full=[fname_full 'table_time_cg_per_iteration_full_'];
  fname_reduced=[fname_reduced 'table_time_cg_per_iteration_'];
elseif tableType==4
  for zz=1:length(kmn_list)
    data_for_table(zz)=sum(total_cg_steps_kmn_all{zz})/length(total_cg_steps_kmn_all{zz});
    data_for_table_sd(zz)=norm(total_cg_steps_kmn_all{zz}-data_for_table(zz),2)/sqrt(length(total_cg_steps_kmn_all{zz}));
  end
  fname_full=[fname_full 'table_total_cg_steps_full_'];
  fname_reduced=[fname_reduced 'table_total_cg_steps_'];
elseif tableType==5
  for zz=1:length(kmn_list)
    prob_gen_times=results{zz}(:,3)-results{zz}(:,2);
    data_for_table(zz)=sum(prob_gen_times)/length(prob_gen_times);
    data_for_table_sd(zz)=norm(prob_gen_times-data_for_table(zz),2)/sqrt(length(prob_gen_times));
  end  
  fname_reduced=[fname_reduced 'table_time_problem_generation_'];
end
data_for_table=data_for_table + sds_to_add*data_for_table_sd;

fname_full=[fname_full sprintf('%s_%s_%s_supp_flag_%d_sds_%d.tex',alg,processor,matrixEnsemble,supp_flag,round(sds_to_add))];
fname_reduced=[fname_reduced sprintf('%s_%s_%s_supp_flag_%d_sds_%d.tex',alg,processor,matrixEnsemble,supp_flag,round(sds_to_add))];

if nargin==7
  fname_full=[fname_full(1:end-4) sprintf('_nonZeros_%d.tex',nonZeros_select)];
  fname_reduced=[fname_reduced(1:end-4) sprintf('_nonZeros_%d.tex',nonZeros_select)];
end
fid_full=fopen(fname_full,'wt');
fid_reduced=fopen(fname_reduced,'wt');


% compute the fits for each value of n separately
n_list=intersect(kmn_list(:,3),kmn_list(:,3));


cap_tableType=cell(4,1);
cap_tableType{1}='time per iteration, excluding support set detection, ';
cap_tableType{2}=sprintf('time for support set detection per iteration with support set flag %d',supp_flag);
cap_tableType{3}='time per cg iteration ';
cap_tableType{4}='total numer of cg steps ';

if sds_to_add~=0
  cap_tableType{tableType} = [cap_tableType{tableType} sprintf('(plus %f standard deviations) ',sds_to_add)];
end



% begin information for table
tmp=sprintf('\\\\begin{center}\n');
fprintf(fid_full,tmp);
tmp=sprintf('\\\\begin{longtable}{|l|l|l|l|l|l|l|}\n');
fprintf(fid_full,tmp);
cap=sprintf('Least squares fits of average %s using the regression model $Const. + \\\\alpha\\\\delta + \\\\beta\\\\rho$. Algorithm %s with matrix ensemble %s using %s code.',cap_tableType{tableType},alg,matrixEnsemble,processor);
if nargin==7
  cap=[cap(1:end-1) sprintf(' with %d nonzeros per column.',nonZeros_select)];
end


tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\\\\\\\\\n',cap,fname_full);
fprintf(fid_full,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid_full,tmp);
tmp=sprintf('n & Const. & $\\\\alpha$ & $\\\\beta$ & $\\\\ell_2$ relative error & $\\\\ell_{\\\\infty}$ relative error & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
fprintf(fid_full,tmp);



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
  
    
  %print n, const, d, r, relative l2, relative l_inf and true l_inf
  %first print all fits to the full table

  
  fprintf(fid_full,'$2^{%d}$ & %0.3g & - & - & %0.3g & %0.3g & %0.3g\\\\\n\\hline\n',log2(n_select),coef1(1),norm(err1./data_for_table(ind),2),norm(err1./data_for_table(ind),inf),norm(err1,inf));
  fprintf(fid_full,'$2^{%d}$ & %0.3g & %0.3g & - & %0.3g & %0.3g & %0.3g\\\\\n\\hline\n',log2(n_select),coef2(1),coef2(2),norm(err2./data_for_table(ind),2),norm(err2./data_for_table(ind),inf),norm(err2,inf));
  fprintf(fid_full,'$2^{%d}$ & %0.3g & - & %0.3g & %0.3g & %0.3g & %0.3g\\\\\n\\hline\n',log2(n_select),coef3(1),coef3(2),norm(err3./data_for_table(ind),2),norm(err3./data_for_table(ind),inf),norm(err3,inf));
  fprintf(fid_full,'$2^{%d}$ & %0.3g & %0.3g & %0.3g & %0.3g & %0.3g & %0.3g\\\\\n\\hline\n',log2(n_select),coef4(1),coef4(2),coef4(3),norm(err4./data_for_table(ind),2),norm(err4./data_for_table(ind),inf),norm(err4,inf));

  %now print the selected fit to the reduced table
  if reduced_flag==1
    table_reduced{j}=sprintf('$2^{%d}$ & %0.3g & - & - & %0.3g\\\\\\\\',log2(n_select),coef1(1),norm(err1,inf));
  elseif reduced_flag==2
    table_reduced{j}=sprintf('$2^{%d}$ & %0.3g & %0.3g & - & %0.3g\\\\\\\\',log2(n_select),coef2(1),coef2(2),norm(err2,inf));
  elseif reduced_flag==3
    table_reduced{j}=sprintf('$2^{%d}$ & %0.3g & - & %0.3g & %0.3g\\\\\\\\',log2(n_select),coef3(1),coef3(2),norm(err3,inf));
  elseif reduced_flag==4
    table_reduced{j}=sprintf('$2^{%d}$ & %0.3g & %0.3g & %0.3g & %0.3g\\\\\\\\',log2(n_select),coef4(1),coef4(2),coef4(3),norm(err4,inf));
  end
  
end



%now need to check the regressions in the reduced table to see if
%any of the variables was never used.  if that is the case, then we
%should remove that variable from the table entirely.
table_reduced_choice=intersect(table_reduced_choice,table_reduced_choice);

%if table_reduced_choice has either choice 4 or both of 2 and 3,
%then we will need to include all options and can just print the
%data in table_reduced directly to the associated reduced fid.



% begin information for table
tmp=sprintf('\\\\begin{table}[h]\n');
fprintf(fid_reduced,tmp);


%tmp=sprintf('\\\\begin{center}\n');
%fprintf(fid_reduced,tmp);




%if table_reduced choice has only option 1 then we need to drop 
%both delta and rho.
if (ismember(1,table_reduced_choice) && length(table_reduced_choice)==1)
  for j=1:length(n_list)
    i1=findstr(table_reduced{j},'- & ');
    i2=[1:1:i1-1 i1+4:1:length(table_reduced{j})];
    table_reduced{j}=table_reduced{j}(i2);
  end
  for j=1:length(n_list)
    i1=findstr(table_reduced{j},'- & ');
    i2=[1:1:i1-1 i1+4:1:length(table_reduced{j})];
    table_reduced{j}=table_reduced{j}(i2);
  end
  cap=sprintf('Least squares fits of average %s using the regression model $Const.$. Algorithm %s with matrix ensemble %s using %s code.',cap_tableType{tableType},alg,matrixEnsemble,processor);
  if nargin==7
    cap=[cap(1:end-1) sprintf(' with %d nonzeros per column.',nonZeros_select)];
  end
  tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\n',cap,fname_reduced);
  fprintf(fid_reduced,tmp);
  
  tmp=sprintf('\\\\begin{center}\n');
  fprintf(fid_reduced,tmp);
  tmp=sprintf('\\\\begin{tabular}{|l|l|l|}\n');
  fprintf(fid_reduced,tmp);
    
  tmp=sprintf('\\\\hline\n');
  fprintf(fid_reduced,tmp);
  tmp=sprintf('n & Const. & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
  fprintf(fid_reduced,tmp);  
  %if table_reduced choice has only option 2 or 3 then we need to
  %drop the one variable not used.
elseif ismember(4,table_reduced_choice)==0 && ( (ismember(2,table_reduced_choice)==0) || (ismember(3,table_reduced_choice)==0) )
    for j=1:length(n_list)
      i1=findstr(table_reduced{j},'- & ');
      i2=[1:1:i1-1 i1+4:1:length(table_reduced{j})];
      table_reduced{j}=table_reduced{j}(i2);
    end
    if ismember(2,table_reduced_choice)==1
      cap=sprintf('Least squares fits of average %s using the regression model $Const. + \\\\alpha\\\\delta$. Algorithm %s with matrix ensemble %s using %s code.',cap_tableType{tableType},alg,matrixEnsemble,processor);
      if nargin==7
        cap=[cap(1:end-1) sprintf(' with %d nonzeros per column.',nonZeros_select)];
      end
      tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\n',cap,fname_reduced);
      fprintf(fid_reduced,tmp);
      
      tmp=sprintf('\\\\begin{center}\n');
      fprintf(fid_reduced,tmp);
      tmp=sprintf('\\\\begin{tabular}{|l|l|l|l|}\n');
      fprintf(fid_reduced,tmp);
      
      tmp=sprintf('\\\\hline\n');
      fprintf(fid_reduced,tmp);
      tmp=sprintf('n & Const. & $\\\\alpha$ & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
      fprintf(fid_reduced,tmp);
    else
      cap=sprintf('Least squares fits of average %s using the regression model $Const. + \\\\beta\\\\rho$. Algorithm %s with matrix ensemble %s using %s code.',cap_tableType{tableType},alg,matrixEnsemble,processor);
      if nargin==7
        cap=[cap(1:end-1) sprintf(' with %d nonzeros per column.',nonZeros_select)];
      end
      tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\n',cap,fname_reduced);
      fprintf(fid_reduced,tmp);
      
      tmp=sprintf('\\\\begin{center}\n');
      fprintf(fid_reduced,tmp);
      tmp=sprintf('\\\\begin{tabular}{|l|l|l|l|}\n');
      fprintf(fid_reduced,tmp);
      
      tmp=sprintf('\\\\hline\n');
      fprintf(fid_reduced,tmp);
      tmp=sprintf('n & Const. & $\\\\beta$ & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
      fprintf(fid_reduced,tmp);
    end
else
  
  cap=sprintf('Least squares fits of average %s using the regression model $Const. + \\\\alpha\\\\delta + \\\\beta\\\\rho$. Algorithm %s with matrix ensemble %s using %s code.',cap_tableType{tableType},alg,matrixEnsemble,processor);
  if nargin==7
    cap=[cap(1:end-1) sprintf(' with %d nonzeros per column.',nonZeros_select)];
  end
  tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\n',cap,fname_reduced);
  fprintf(fid_reduced,tmp);
  
  tmp=sprintf('\\\\begin{center}\n');
  fprintf(fid_reduced,tmp);
  tmp=sprintf('\\\\begin{tabular}{|l|l|l|l|l|}\n');
  fprintf(fid_reduced,tmp);
  
  tmp=sprintf('\\\\hline\n');
  fprintf(fid_reduced,tmp);
  tmp=sprintf('n & Const. & $\\\\alpha$ & $\\\\beta$ & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
  fprintf(fid_reduced,tmp);
end


%now print the data in table_reduced
%warning('off','all');
for zz=1:length(n_list)
  fprintf(fid_reduced,table_reduced{zz});
  fprintf(fid_reduced,'\n');
  fprintf(fid_reduced,'\\hline');
  fprintf(fid_reduced,'\n');
end
%warning('on','all');


  
tmp=sprintf('\\\\end{longtable}\n');
fprintf(fid_full,tmp);
tmp=sprintf('\\\\end{center}\n');
fprintf(fid_full,tmp);


tmp=sprintf('\\\\end{tabular}\n');
fprintf(fid_reduced,tmp);
tmp=sprintf('\\\\end{center}\n');
fprintf(fid_reduced,tmp);
tmp=sprintf('\\\\end{table}\n');
fprintf(fid_reduced,tmp);

fclose(fid_full);
fclose(fid_reduced);



