function make_table_PCGACS(alg_list, ens, nz, noise_level, vecDistr, kmn_list)

% kmn_list is an optional argument

% in case there are too many tests, this is the number of tests we want in the table 
tests_per_k = 100;   % for PCGACS this is 1000; 

format shortg
% this script makes a table of a few select large number of tests

m_minimum=100;
% only present results where m is at least m_minimum

% create string to access noise data
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

    if (noise_level == 0)
      noise_string = '';
    end

% replace '.' with '-' for the output filenames for the plots
fname_noise_string = strrep(noise_string,'.','-');

vec_str = ['_vecDistr' num2str(vecDistr)];

% look for large numbers of tests at values of delta near this
delta_refined=[0.01 0.05 0.1 0.3 0.6];

% find tests for values of rho that are these fractions of the
% largest phase transition for the algorithms presented
rho_multiple_list=[0.5 0.75 0.9];


delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
delta_list=sort(delta_list);

tmp=[];
for j=1:length(delta_refined)
  [junk,ind]=sort(abs(delta_list-delta_refined(j)),'ascend');
  tmp=[tmp; delta_list(ind(1))];
end

clear delta_refined;
delta_list=tmp;

% smv has an offset of 2 in the column number, so iterations is in
% column 13 rather than 11.
% gen has an offset of 1 in the column number, so iterations is in
% column 12 rather than 11.

if strcmp(ens,'dct')
  n_select=2^18;
  ens_offset=0;
elseif strcmp(ens,'gen')
  n_select=2^12;
  ens_offset=1;
elseif strcmp(ens,'smv')
  n_select=2^16;
  nz_select=7;
  ens_offset=2;
else
  error('ens should be dct, smv, or gen');
end



rho_star_list=zeros(size(delta_list));



%{
for j=1:length(alg_list)
  fname=['results_' alg_list{j} '_S_' ens vec_str fname_noise_string '.mat']
  load(fname);
whos

  ind=find(n_list==n_select);
  if strcmp(ens,'smv')
    ind1=find(nz_list==nz_select);
    ind=intersect(ind,ind1);
  end
  betas=betas{ind};
  deltas=deltas{ind};

  tmp=zeros(size(delta_list));
  for jj=1:length(delta_list)
    [junk,ind]=sort(abs(deltas-delta_list(jj)),'ascend');
    tmp(jj)=1./betas(ind(1),2);
  end
  rho_star_list=max(rho_star_list,tmp);
end
%}

if nargin<4
% no kmn_list was provided so we
% need to determine the grid of k,m,n where the algorithms were tested
  kmn_list=[];
  for j=1:length(delta_list)
    m=ceil(delta_list(j)*n_select);
    if m>=m_minimum
      k=[];
      for jj=1:length(rho_multiple_list)
        r=rho_star_list(j)*rho_multiple_list(jj);
        k=[k; ceil(r*m)];
      end
      kmn_list=[kmn_list; [k m*ones(size(k)) n_select*ones(size(k))]];
    end
  end
  kmn_list=intersect(kmn_list,kmn_list,'rows');
end % end find kmn_list if none was provided

%kmn_list
%ind=find(results(:,3)==n_select);
%r=results(ind,:);
%c=intersect(r(:,1:3),r(:,1:3),'rows');
%num=zeros(length(c),1);
%for j=1:length(num)
%  k=c(j,1);
%  m=c(j,2);
%  ind=find(r(:,1)==k & r(:,2)==m);
%  num(j)=length(ind);
%end
%[junk,ind]=sort(num,'descend');
%[c(ind(1:50),1:3) num(ind(1:50))]
%pause


fname_save=['table_' ens ];

if strcmp(ens,'dct')==1
  if noise_level==0
    ProbClass = sprintf('$(DCT,B)$');
  else 
    ProbClass = sprintf('$(DCT,B_\\\\epsilon)$ for $\\\\epsilon=1/10$');
  end
elseif strcmp(ens,'gen')==1
  if noise_level==0
    ProbClass = sprintf('$({\\\\cal N},B)$');
  else
    ProbClass = sprintf('$({\\\\cal N},B_\\\\epsilon)$ for $\\\\epsilon=1/10$');
  end
elseif strcmp(ens,'smv')==1
  if noise_level==0
    ProbClass = sprintf('$({\\\\cal S}_%d,B)$',nz_select);
  else
    ProbClass = sprintf('$({\\\\cal S}_%d,B_\\\\epsilon)$ for $\\\\epsilon=1/10$',nz_select);
  end
  fname_save=[fname_save '_' num2str(nz_select) 'nz'];
else
  error('ens must be: dct, gen, or smv')
end

cap=sprintf('Average and standard deviation of performance characteristics for NIHT, HTP, and CSMPSP for %s with $n=2^{%d}$',ProbClass,log2(n_select));

fname_save = [fname_save '_' num2str(n_select) vec_str fname_noise_string '.tex'];

fid=fopen(fname_save,'wt');


tmp=sprintf('\\\\begin{center}\n');
fprintf(fid,tmp);
tmp=sprintf('\\\\begin{table}\n');
fprintf(fid,tmp);

tmp=sprintf('{\\\\tiny\n');
fprintf(fid,tmp);

tmp=sprintf('\\\\begin{center}\n');
fprintf(fid,tmp);


if noise_level==0
  tmp=sprintf(' \\\\begin{tabular}{|l|l|l|r|l|l|l|l|l|}\n');
  fprintf(fid,tmp);
else
  tmp=sprintf(' \\\\begin{tabular}{|l|l|l|r|l|l|l|l|}\n');
  fprintf(fid,tmp);
end
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);

if noise_level == 0
  tmp=['\\multirow{2}{*}{$m$} & \\multirow{2}{*}{$k$} & \\multirow{2}{*}{Algorithm} & successes / & \\multicolumn{2}{c|}{fraction of correct support} & time (ms) & iterations & convergence rate \\\\\n'];
  fprintf(fid,tmp);
  tmp=[' & &  & ' num2str(tests_per_k) ' tests & successes & failures & successes & successes & successes \\\\\n'];
  fprintf(fid,tmp);
else
  tmp=['\\multirow{2}{*}{$m$} & \\multirow{2}{*}{$k$} & \\multirow{2}{*}{Algorithm} & successes / & \\multicolumn{2}{c|}{fraction of correct support} & time (ms) & iterations  \\\\\n'];
  fprintf(fid,tmp);
  tmp=[' & &  & 1000 tests & successes & failures & successes & successes \\\\\n'];
  fprintf(fid,tmp);
end

tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
%tmp=sprintf('\\\\hline\n');
%fprintf(fid,tmp);

Tol=10^(-4);

% for printing purposes need to know how many of each m there are
m_list=intersect(kmn_list(:,2),kmn_list(:,2));
num_m_list=zeros(size(m_list));
for j=1:length(m_list)
  num_m_list(j)=sum(kmn_list(:,2)==m_list(j));
end
m_ind=1; 

for j=1:size(kmn_list,1)
  
  num_tests=zeros(length(alg_list),1);
  success=zeros(length(alg_list),1);
  time=zeros(length(alg_list),1);
  time_sd=zeros(length(alg_list),1);
  iter=zeros(length(alg_list),1);
  iter_sd=zeros(length(alg_list),1);
  conv=zeros(length(alg_list),1);
  conv_sd=zeros(length(alg_list),1);
  succ_supp=zeros(length(alg_list),1);
  succ_supp_sd=zeros(length(alg_list),1);
  fail_supp=zeros(length(alg_list),1);
  fail_supp_sd=zeros(length(alg_list),1);
  
  % select the values of k near the tests
  k=kmn_list(j,1);
  m=kmn_list(j,2);
  n=kmn_list(j,3);
  
  if k<10
    k_gap=0;
  elseif k<2000
    k_gap=1;
  else 
    k_gap=3;
  end
  
  for jj=1:length(alg_list)  
  
    fname=['results_' alg_list{jj} '_S_' ens vec_str noise_string '.mat'];
    load(fname);
    
    % select the values of k near the tests
  
    ind1=find(abs(k-results(:,1)) <= k_gap);
    intersect(results(ind1,1),results(ind1,1));
    ind2=find(kmn_list(j,2)==results(:,2));
    ind3=find(kmn_list(j,3)==results(:,3));

    ind=intersect(ind1,ind2);
    ind=intersect(ind,ind3);
    if strcmp(ens,'smv')
      ind1=find(results(:,5)==nz_select);
      ind=intersect(ind,ind1);
    end
    
    results_selected=results(ind,:);
    
    % remove any NaN or Inf
    ind1=find(sum(isnan(results_selected),2)==0);
    ind2=find(sum(isinf(results_selected),2)==0);
    ind=intersect(ind1,ind2);
    results_selected=results_selected(ind,:);
    results_selected(:,12+ens_offset)=min(1,results_selected(:,12+ens_offset));
 

    num_tests(jj)=size(results_selected,1);
    if num_tests(jj) > tests_per_k
      sampleind = randperm(num_tests(jj));
      results_selected=results_selected(sampleind(1:tests_per_k),:);
      num_tests(jj)=size(results_selected,1);
    end  

    if noise_level == 0
      successind=find(results_selected(:,7+ens_offset)<10*Tol);
      failind=find(results_selected(:,7+ens_offset)>=10*Tol);
    else
      successind=find(results_selected(:,6+ens_offset)<10*Tol+2*noise_level);
      failind=find(results_selected(:,6+ens_offset)>=10*Tol+2*noise_level);
    end
    numsuccess=length(successind);
    numfail=length(failind);

    success(jj)=numsuccess;
    time(jj)=sum(results_selected(successind,8+ens_offset))/numsuccess;
    time_sd(jj)=norm(results_selected(successind,8+ens_offset)-time(jj),2)/sqrt(numsuccess-1);
    iter(jj)=sum(results_selected(successind,11+ens_offset))/numsuccess;
    iter_sd(jj)=norm(results_selected(successind,11+ens_offset)-iter(jj),2)/sqrt(numsuccess-1);
    conv(jj)=sum(results_selected(successind,12+ens_offset))/numsuccess;
    conv_sd(jj)=norm(results_selected(successind,12+ens_offset)-conv(jj),2)/sqrt(numsuccess-1);
    succ_supp(jj)=sum(results_selected(successind,13+ens_offset)./results_selected(successind,1))/numsuccess;
    succ_supp_sd(jj)=norm(results_selected(successind,13+ens_offset)./results_selected(successind,1)-succ_supp(jj),2)/sqrt(numsuccess-1);
    fail_supp(jj)=sum(results_selected(failind,13+ens_offset)./results_selected(failind,1))/numfail;
    fail_supp_sd(jj)=norm(results_selected(failind,13+ens_offset)./results_selected(failind,1)-fail_supp(jj),2)/sqrt(numfail-1);
  
    if (numsuccess==1)
      time_sd(jj)=0;
      iter_sd(jj)=0;
      conv_sd(jj)=0;
      succ_supp_sd(jj)=0;
    end
    if (numfail==1)
      fail_supp_sd(jj)=0;
    end
%    supp(jj)=sum(results_selected(:,13+ens_offset))/num_tests(jj);
%    supp_sd(jj)=norm(results_selected(:,13+ens_offset)-supp(jj),2)/sqrt(num_tests(jj)-1);
    
%  end
  
  % now need to put the data in a table
%  for jj=1:length(alg_list)

    % determine if k and m should be printed

    m_print=0;
    k_print=0;
    if j==1 & jj==1
      m_print=1;
    end
    if ( (jj==1) & (j>1) & (kmn_list(j,2)~=kmn_list(j-1,2)) ) % m changed
      m_print=1;
    end
    if jj==1
      k_print=1;
    end
    


    if m_print==1 & k_print==1
      tmp=sprintf(' \\\\hline\n');
      fprintf(fid,tmp);
    elseif m_print==0 & k_print==1
      if noise_level == 0
        tmp=sprintf(' \\\\cline{2-9}\n');
      else 
        tmp=sprintf(' \\\\cline{2-8}\n');
      end
      fprintf(fid,tmp);
    else
      if noise_level == 0
        tmp=sprintf(' \\\\cline{3-9}\n');
      else
        tmp=sprintf(' \\\\cline{3-8}\n');
      end
      fprintf(fid,tmp);
    end
    
    
    tmp=[];
    if m_print==1
      tmp=strcat(tmp,['\\multirow{' num2str(num_m_list(m_ind)*length(alg_list)) '}{*}{' num2str(kmn_list(j,2)) '} &']);
      m_ind=m_ind+1;
    else
      tmp=[ ' & '];
    end
    if k_print==1
      tmp=strcat(tmp,[' \\multirow{' num2str(length(alg_list)) '}{*}{' num2str(kmn_list(j,1)) '} &']);
    else
      tmp=strcat(tmp,[ ' &']);
    end
  %{  
%    tmp2=sprintf(' %s & %d/%d & %0.4g($\\\\pm$ %0.1g) & %0.4g($\\\\pm$ %0.1g) & %0.4g($\\\\pm$ %0.1g) & %0.4g($\\\\pm$ %0.1g)',alg_list{jj},success(jj),num_tests(jj),supp(jj),supp_sd(jj),time(jj),time_sd(jj),iter(jj),iter_sd(jj),conv(jj),conv_sd(jj));

    if noise_level==0
      tmp2=sprintf(' %s & %d & %0.3g($\\\\pm$ %0.3g) & %0.3g($\\\\pm$ %0.3g) & %0.1f($\\\\pm$ %0.1f) & %d($\\\\pm$ %0.1g) & %0.3f($\\\\pm$ %0.3f)',alg_list{jj},success(jj),succ_supp(jj),succ_supp_sd(jj),fail_supp(jj),fail_supp_sd(jj),time(jj),time_sd(jj),round(iter(jj)),iter_sd(jj),conv(jj),conv_sd(jj));
    else
      tmp2=sprintf(' %s & %d & %0.3g($\\\\pm$ %0.3g) & %0.3g($\\\\pm$ %0.3g) & %0.1f($\\\\pm$ %0.1f) & %d($\\\\pm$ %0.1g)',alg_list{jj},success(jj),succ_supp(jj),succ_supp_sd(jj),fail_supp(jj),fail_supp_sd(jj),time(jj),time_sd(jj),round(iter(jj)),iter_sd(jj));
    end
%}
    alg_succ_str = sprintf(' %s & %d &', alg_list{jj}, success(jj));
    time_iter_str = sprintf(' %0.1f($\\\\pm$ %0.1f) & %d($\\\\pm$ %0.1f) ', time(jj),time_sd(jj),round(iter(jj)),iter_sd(jj));
    conv_str = sprintf(' & %0.3f($\\\\pm$ %0.3f)', conv(jj),conv_sd(jj)); 

    if numsuccess == 0
      supp_frac_str=sprintf(' -- & %0.3f($\\\\pm$ %0.3f) &',fail_supp(jj),fail_supp_sd(jj));
      time_iter_str = sprintf(' -- & -- ');
      conv_str = sprintf(' & -- '); 
    else 
      if numfail == 0
        if (succ_supp(jj)>.9995)
          supp_frac_str=sprintf(' %0.3g($\\\\pm$ %0.3g) & -- &',succ_supp(jj),succ_supp_sd(jj));
        else
          supp_frac_str=sprintf(' %0.3f($\\\\pm$ %0.3f) & -- &',succ_supp(jj),succ_supp_sd(jj));
        end
      else
        if (succ_supp(jj)>.9995)
          supp_frac_str=sprintf(' %0.3g($\\\\pm$ %0.3g) & %0.3f($\\\\pm$ %0.3f) &',succ_supp(jj),succ_supp_sd(jj),fail_supp(jj),fail_supp_sd(jj));
        else
          supp_frac_str=sprintf(' %0.3f($\\\\pm$ %0.3f) & %0.3f($\\\\pm$ %0.3f) &',succ_supp(jj),succ_supp_sd(jj),fail_supp(jj),fail_supp_sd(jj));
        end
      end
    end

    if noise_level==0
      tmp2=sprintf(' %s %s %s %s ', alg_succ_str,supp_frac_str,time_iter_str,conv_str);
    else
      tmp2=sprintf(' %s %s %s ', alg_succ_str,supp_frac_str,time_iter_str);
    end


    tmp=strcat(tmp,tmp2);
    tmp2=' \\\\ \n';
    tmp=strcat(tmp,tmp2);
    fprintf(fid,tmp);
    
  end
  
%  [num_tests success time time_sd iter iter_sd conv conv_sd supp supp_sd]
%  kmn_list(j,:)
%  pause
  
end

tmp=sprintf(' \\\\hline\n');
fprintf(fid,tmp);
      
tmp=sprintf('\\\\end{tabular}\n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{center}\n');
fprintf(fid,tmp);

tmp=sprintf('}\n');
fprintf(fid,tmp);

tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\n',cap,fname_save(1:end-4));
fprintf(fid,tmp);
tmp=sprintf('\\\\end{table}\n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{center}\n');
fprintf(fid,tmp);

fclose(fid);



