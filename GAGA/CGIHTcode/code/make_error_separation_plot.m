% This script takes all tests for the problem class (Mat,B)
% and plots the l_infty error against iterations.  This demonstrates
% a clear separation between success and failure validating the stopping criteria.


% choose to save all files as .eps with eps_flag (1) on, (0) off
eps_flag = 0;
% default prints figures as .pdf
if eps_flag
  fig_suffix = '.eps';
  print_command = '-depsc';
else
  fig_suffix = '.pdf';
  print_command = '-dpdf';
end

fnames_all=ls('results_*.mat');

fname_start='results';
fname_end='.mat';

ind_start=strfind(fnames_all,fname_start);
ind_end=strfind(fnames_all,fname_end)+3;

fnames=cell(length(ind_start),1);
for j=1:length(ind_start)
  fnames{j}=fnames_all(ind_start(j):ind_end(j));
end

% keep only names that also include 'NIHT','HTP','CSMPSP'
fnames_reduced=cell(length(ind_start),1);
k=1;
for j=1:length(ind_start)
  ind=strfind(fnames{j},'NIHT');
  ind=[ind strfind(fnames{j},'HTP')];
%  ind=[ind strfind(fnames{j},'IHT')];
  ind=[ind strfind(fnames{j},'CSMPSP')];
  if isempty(ind)==0
    fnames_reduced{k}=fnames{j};
    k=k+1;
  end
end
k=k-1;

fnames=cell(k,1);
for j=1:k
  fnames{j}=fnames_reduced{j};
end


col_iter=zeros(length(fnames),1);
col_infty=zeros(length(fnames),1);
num_tests=zeros(length(fnames),1);

for j=1:length(fnames)
  load(fnames{j})
  ind_infinity=strfind(columns,'infinity');
  ind_iterations=strfind(columns,'iterations,');
  ind_commas=strfind(columns,',');
  col_infty(j)=sum(ind_commas<ind_infinity)+1;
  col_iter(j)=sum(ind_commas<ind_iterations)+1;
%  columns
%  col_infty(j)
%  col_iter(j)
%  pause
end

max_iter=0;

%num_between=0;
hold off
for j=1:length(fnames)
  load(fnames{j})
  num_tests(j)=size(results,1);
  semilogx(results(:,col_infty(j)),results(:,col_iter(j)),'o')
  max_iter=max(max_iter,max(results(:,col_iter(j))));

%  ind1=find(results(:,col_infty(j))>10^(-3));
%  ind2=find(results(:,col_infty(j))<10^(-2));
  
%  num_between=num_between+length(intersect(ind1,ind2))
  
%  fnames{j}
%  num_tests(j)
  hold on
%  pause
end
x=10^(-2)*ones(2,1);
y=[0;max_iter];
plot(x,y,'r-','linewidth',3)
axis([10^(-8),10^4,0,max_iter])
xlabel('$\|x-\hat{x}\|_{\infty}$','interpreter','latex','fontsize',14)
ylabel('Iterations','fontsize',14)
title('Seperation of successful and unsuccesful problem instances','fontsize',14)
display(sprintf('Error separation plot includes %0.3g problem instances',sum(num_tests)))

fname_save = ['plots/error_separation_plot' fig_suffix];
print(fname_save,print_command)

% fname_save='error_separation_caption.tex';
% fid=fopen(fname_save,'wt');

% tmp=sprintf('\\\\caption{Separation plot of the error for all %0.2g problem instances presented here for algorithms NIHT, HTP, CSMPSP and problem class $Mat\\\\in\\\\{\\\\cal B\\\}$ with $Mat\\\\{\\{\\cal B},{\cal S}_7,DCT\\\\}$',sum(num_tests))
% fclose(fid)

