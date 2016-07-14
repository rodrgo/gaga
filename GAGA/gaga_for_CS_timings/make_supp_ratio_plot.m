function make_supp_ratio_plot(alg,matrixEnsemble,n_select,nonZeros_select)

% alg will be either: NIHT or HTP
% matrixEnsemble will be either: dct, smv, or gen
% n_select is the number of columns
% nonZeros_select is only for matrixEnsemble smv and is the number of
% nonzeros per column

% makes plots of the supp_set_timings and their ratios for n=n_select




% the other variants are always one choice in the data considered here
vecDistribution=1; %0 is for uniform [0,1], 
                   %1 is for random {-1,1},
                   %2 is for gaussian.


% supp_flag is: 0 for dynamic binning, 1 is for always binning,
%               2 for sort with dynamics, and 3 is for always sorting

supp_flag_list=[0 1 2 3]; 

kmn=cell(4,1);
times_all=cell(4,1);
times_ave=cell(4,1);

for qq=1:length(supp_flag_list)
  supp_flag=supp_flag_list(qq);
  fname_save=sprintf('results_%s_%s_timing_supp_flag_%d',alg,matrixEnsemble,supp_flag);
  load(fname_save)
  

  %USE NARGIN FOR NONZEROS
  % take the data with the specified n_select, and for m>100 (where m
  % is the number of rows), and if nonZeros is specified then select
  % only that portion of the data for smv.
  if nargin==4
    ind=find(kmnp_list(:,4)==nonZeros_select);
    kmn_list=kmnp_list(ind,1:3);
    time_supp_set_kmn_all=time_supp_set_kmnp_all(ind);
  end
  ind=find(kmn_list(:,3)==n_select);
  kmn{qq}=kmn_list(ind,:);
  times_all{qq}=time_supp_set_kmn_all(ind);
  times_ave{qq}=zeros(length(kmn{qq}),1);
  for jj=1:length(kmn{qq})
    times_ave{qq}(jj)=sum(times_all{qq}{jj})/length(times_all{qq}{jj});
  end
%  plot(times_ave{qq})
%  qq
%  pause
end

% compare 0 & 1, and 0 & 2. (the 2 & 3 is easy to see, and the 0 to
% 3 is silly as 3 of obviously worse than 2.)


%************************************************
% The data is now ready, and tables can be formed
%



fname_full='plots/plot_time_supp_set_';


%compute the average and standard deviation of the measured quantities


fname_full=[fname_full sprintf('%s_gpu_%s_supp_flags_%d.tex',alg,matrixEnsemble,n_select)];
if nargin==4
  fname_full=[fname_full(1:end-4) sprintf('_nonZeros_%d.tex',nonZeros_select)];
end
fid_full=fopen(fname_full,'wt');


cap=sprintf('time (in \\\\underline{milliseconds}) for support set detection per iteration with $n=2^{%d}$',log2(n_select));


% begin information for figure
tmp=sprintf('\\\\begin{center}\n');
fprintf(fid_full,tmp);
tmp=sprintf('\\\\begin{figure}\n');
fprintf(fid_full,tmp);
tmp=sprintf('\\\\begin{tabular}{cc}\n');
fprintf(fid_full,tmp);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SOME INCLUDE GRAPHICS SHOULD BE INSERTED HERE!!!!!!!!!!

% MAKE A TABLE OF FIGURES USING TWO COLUMNS AND UP TO 3 ROWS,
% SHOWING THE LARGEST 6 VALUES OF n WE HAVE DATA FOR.



small_m=100;


for j=1:length(supp_flag_list)

  ind=find(kmn{j}(:,2)>small_m); 

  d=kmn{j}(ind,2)/n_select;
  r=kmn{j}(ind,1)./kmn{j}(ind,2);
  m_list=intersect(kmn{j}(ind,2),kmn{j}(ind,2));
  %keep only those m that give delta>delta_min
  delta_min=0.01;
  ind_tmp=find( (m_list/n_select) > delta_min );
  m_list=m_list(ind_tmp);
  
  [r_mesh,d_mesh]=meshgrid(m_list/n_select,min(r):0.01:max(r));
  F=TriScatteredInterp(r,d,times_ave{j}(ind),'natural');
  data_for_table_mesh=F(d_mesh,r_mesh);
  for zz=1:length(m_list)
    ind=find(kmn_list(:,2)==m_list(zz));
    rmax=max(kmn_list(ind,1))/m_list(zz);
    if zz>1
      ind_old=find(kmn_list(:,2)==m_list(zz-1));
      rmax_old=max(kmn_list(ind_old,1))/m_list(zz-1);
      rmax=min(rmax,rmax_old);
    end
    for qq=1:size(data_for_table_mesh,1)
      if d_mesh(qq,zz)>rmax-0.05 %NOTE THE 0.9 WHICH IS TO REMOVE
                                %THE DATA NEAREST THE PHASE TRANSITION.
        data_for_table_mesh(qq,zz)=0;
      end
    end
  end

  mesh(r_mesh,d_mesh,data_for_table_mesh); 
  view([-18 26])
  colorbar 
  figname=[fname_full(1:end-4) sprintf('_supp_flag_%d_%d.pdf',supp_flag_list(j),n_select)];
  print('-dpdf',figname)
  
  tmp=sprintf('\\\\includegraphics[height=2.3in,width=2.15in,trim=1in 3in 1in 2.6in]{%s}\n',figname(1:end));
  fprintf(fid_full,tmp);
  
  if mod(j,2)
    tmp=sprintf(' & ');
    fprintf(fid_full,tmp);
    tmp=sprintf('\n');
    fprintf(fid_full,tmp);
  else
    tmp=sprintf(' \\\\\\\\');
    fprintf(fid_full,tmp);
    tmp=sprintf('\n');
    fprintf(fid_full,tmp);
    if j==2
      tmp=sprintf('(a) & (b) \\\\\\\\');
      fprintf(fid_full,tmp);
      tmp=sprintf('\n');
      fprintf(fid_full,tmp);
    elseif j==4
      tmp=sprintf('(c) & (d) \\\\\\\\');
      fprintf(fid_full,tmp);
      tmp=sprintf('\n');
      fprintf(fid_full,tmp);
    end
  end
  
    
end


% THE FIRST RATIO PLOT
sup1=1;
sup2=2;

[kmn_ratio,ind1,ind2]=intersect(kmn{sup1},kmn{sup2},'rows');
times_ave_ratio=times_ave{sup1}(ind1)./times_ave{sup2}(ind2);

%plot(times_ave_ratio)

ind=find(kmn_ratio(:,2)>small_m); 

d=kmn_ratio(ind,2)/n_select;
r=kmn_ratio(ind,1)./kmn_ratio(ind,2);
m_list=intersect(kmn_ratio(ind,2),kmn_ratio(ind,2));
%keep only those m that give delta>delta_min
delta_min=0.05;
ind_tmp=find( (m_list/n_select) > delta_min );
m_list=m_list(ind_tmp);
  
[r_mesh,d_mesh]=meshgrid(m_list/n_select,min(r):0.01:max(r));
F=TriScatteredInterp(r,d,times_ave_ratio(ind),'natural');
data_for_table_mesh=F(d_mesh,r_mesh);
for zz=1:length(m_list)
  ind=find(kmn_list(:,2)==m_list(zz));
  rmax=max(kmn_list(ind,1))/m_list(zz);
  if zz>1
    ind_old=find(kmn_list(:,2)==m_list(zz-1));
    rmax_old=max(kmn_list(ind_old,1))/m_list(zz-1);
    rmax=min(rmax,rmax_old);
  end
  for qq=1:size(data_for_table_mesh,1)
    if d_mesh(qq,zz)>rmax-0.05 %NOTE THE 0.9 WHICH IS TO REMOVE
                              %THE DATA NEAREST THE PHASE
                              %TRANSITION
      data_for_table_mesh(qq,zz)=0;
    end
  end
end

clist=[0.1 0.2 0.4 0.6 0.8 1];
[c, h]=contour(r_mesh,d_mesh,data_for_table_mesh,clist); 
clabel(c,h)
%view([-18 26])
colorbar 
figname=[fname_full(1:end-4) sprintf('_supp_flags_%d_and_%d_%d.pdf',supp_flag_list(sup1),supp_flag_list(sup2),n_select)];
print('-dpdf',figname)
  
tmp=sprintf('\\\\includegraphics[height=2.3in,width=2.15in,trim=1in 3in 1in 2.6in]{%s}\n',figname(1:end));
fprintf(fid_full,tmp);

tmp=sprintf(' & ');
fprintf(fid_full,tmp);
tmp=sprintf('\n');
fprintf(fid_full,tmp);



% THE SECOND RATIO PLOT
sup1=1;
sup2=3;

[kmn_ratio,ind1,ind2]=intersect(kmn{sup1},kmn{sup2},'rows');
times_ave_ratio=times_ave{sup1}(ind1)./times_ave{sup2}(ind2);

%plot(times_ave_ratio)

ind=find(kmn_ratio(:,2)>small_m); 

d=kmn_ratio(ind,2)/n_select;
r=kmn_ratio(ind,1)./kmn_ratio(ind,2);
m_list=intersect(kmn_ratio(ind,2),kmn_ratio(ind,2));
%keep only those m that give delta>delta_min
delta_min=0.05;
ind_tmp=find( (m_list/n_select) > delta_min );
m_list=m_list(ind_tmp);
  
[r_mesh,d_mesh]=meshgrid(m_list/n_select,min(r):0.01:max(r));
F=TriScatteredInterp(r,d,times_ave_ratio(ind),'natural');
data_for_table_mesh=F(d_mesh,r_mesh);
for zz=1:length(m_list)
  ind=find(kmn_list(:,2)==m_list(zz));
  rmax=max(kmn_list(ind,1))/m_list(zz);
  if zz>1
    ind_old=find(kmn_list(:,2)==m_list(zz-1));
    rmax_old=max(kmn_list(ind_old,1))/m_list(zz-1);
    rmax=min(rmax,rmax_old);
  end
  for qq=1:size(data_for_table_mesh,1)
    if d_mesh(qq,zz)>rmax-0.05 %NOTE THE 0.9 WHICH IS TO REMOVE
                              %THE DATA NEAREST THE PHASE
                              %TRANSITION
      data_for_table_mesh(qq,zz)=0;
    end
  end
end

clist=[0.2 0.4 0.6 0.8 1 1.5 2 2.5 3 3.5 5];
[c, h]=contour(r_mesh,d_mesh,data_for_table_mesh,clist); 
clabel(c,h)
%view([-18 26])
colorbar 
figname=[fname_full(1:end-4) sprintf('_supp_flags_%d_and_%d_%d.pdf',supp_flag_list(sup1),supp_flag_list(sup2),n_select)];
print('-dpdf',figname)
  
tmp=sprintf('\\\\includegraphics[height=2.3in,width=2.15in,trim=1in 3in 1in 2.6in]{%s}\n',figname(1:end));
fprintf(fid_full,tmp);


      tmp=sprintf('\\\\\\\\');
      fprintf(fid_full,tmp);
      tmp=sprintf('\n');
      fprintf(fid_full,tmp);

      tmp=sprintf('(e) & (f)');
      fprintf(fid_full,tmp);
      tmp=sprintf('\n');
      fprintf(fid_full,tmp);






tmp=sprintf('\\\\end{tabular}\n');
fprintf(fid_full,tmp);

% supp_flag is: 0 for dynamic binning, 1 is for always binning,
%               2 for sort with dynamics, and 3 is for always sorting


cap=sprintf('Plot of average %s. The GPU implementation of %s with the %s matrix ensemble.  Panel (a) is $\\\\fsupp$. Panel (b) uses $\\\\fsupp$ with minValue set to maxValue to force the support set identification at each call.  Panel (c) uses sorting but only when the update is sufficiently large that the support set could have changed.  Panel (d) uses sorting at each iteration. Panel (e) is the ratio of the times in Panel (a) over those in Panel (b).  Panel (f) is the ratio of the times in Panel (a) over those in Panel (c).',cap,alg,matrixEnsemble);
if nargin==4
  cap=[cap(1:end-1) sprintf(' with %d nonzeros per column.',nonZeros_select)];
end

label=['fig:plot_supp_' alg '_' matrixEnsemble '_' num2str(log2(n_select))];
tmp=sprintf('\\\\caption{%s}\\\\label{%s}\\\\\n',cap,label);
fprintf(fid_full,tmp);

tmp=sprintf('\\\\end{figure}\n');
fprintf(fid_full,tmp);
tmp=sprintf('\\\\end{center}\n');
fprintf(fid_full,tmp);

fclose(fid_full);






