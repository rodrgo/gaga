function [km_list best_alg time_ave_list error_ave_list] = make_best_alg_plot(ensemble,n,nonzeros,noise_level,vecDistr,alg_list)

tic

%addpath ../

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

% no_zero_noise_str = 1 means that files do not contain '_noise0.000.mat'
% so don't look for this.  On the other hand, no_zero_noise_str=0 means you should look for it.
no_zero_noise_str = 1;

% alg_list must contain at least 2 and no more than 6 algorithms
if nargin<6
  alg_list=cell(5,1);
  alg_list{1}='CGIHTrestarted';
  alg_list{2}='CGIHT';
  alg_list{3}='NIHT';
  alg_list{4}='FIHT';
  alg_list{5}='CSMPSP';
  if nargin<5
    vecDistr=1;
    if nargin<4
      noise_level=0;
    end
  end
end

% label_list, c_list, s_list should be defined to label algorithms in selection plots.
% For consistency across papers, NIHT should be a black circle and CSMPSP a blue square
label_list=cell(length(alg_list),1);
c_list='krbgmy';
s_list='o+sxd*';
for i=1:length(alg_list)
  alg=alg_list{i};
  if strcmp(alg,'NIHT')
    label_list{i} = ['NIHT: circle'];
    c_list(i)='k';
    s_list(i)='o';
  elseif strcmp(alg,'CSMPSP')
    label_list{i} = ['CSMPSP: square'];
    c_list(i)='b';
    s_list(i)='s';  
  elseif strcmp(alg,'CGIHTrestarted')
    label_list{i} = ['CGIHTrestarted: plus'];
    c_list(i)='r';
    s_list(i)='+';
  elseif strcmp(alg,'CGIHT')
    label_list{i} = ['CGIHT: diamond'];
    c_list(i)='m';
    s_list(i)='d';  
  elseif strcmp(alg,'HTP')
    label_list{i} = ['HTP: plus'];       % for PCGACS
    c_list(i)='r';                       % for PCGACS
    s_list(i)='+';                       % for PCGACS
    %label_list{i} = ['HTP: asterisk'];  % for CGIHT paper
    %c_list(i)='r';                      % for CGIHT paper
    %s_list(i)='*';                      % for CGIHT paper
  elseif strcmp(alg,'FIHT')
    label_list{i} = ['FIHT: times'];
    c_list(i)='g';
    s_list(i)='x';
  else 
    label_list{i} = [alg_list{i} ': asterisk'];
    c_list(i)='y';
    s_list(i)='*';
  end
end

% minappear is the minimum number of times a symbol must appear
% in the selection map in order to be labeled
minappear = 1;

% to use l_infity criteria, set l_inf_success = 1
% to use l_two criteria (for noise), set l_inf_success = 0
l_inf_success = 1;
if l_inf_success
  succCriteria = 'l_infinity';
  error_tol = 0.01;
else
  succCriteria = 'l_two';
  error_tol=.01+2*noise_level;
end

% To allow for possible smoothing in the selection of best 
% algorithm, which requires at least a 1/20 change 
% set smoothing = 1.
smoothing = 0;

% set font sizes in the plots
% labelfont is for the axis labels, plotfont for text in the plot, titlefont for the title
labelfont = 16;
plotfont = 16;
titlefont = 16;

ens=ensemble;

if strcmp(ens,'smv')
  nz = nonzeros;
else
  nz = 0;
end

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

if no_zero_noise_str
    if (noise_level == 0)
      noise_string = '';
    end
end

% replace '.' with '-' for the output filenames for the plots
fname_noise_string = strrep(noise_string,'.','-');

vec_str = ''; %['_vecDistr' num2str(vecDistr)];


% This defines a string VV for writing the problem class
% as (MM,VV) based on vecDistribution
switch vecDistr
  case 0
    VV='U';
  case 1
    VV='B';
  case 2
    VV='N';
  otherwise
    VV='VV';
    warning('vecDistribution was not from the list 0,1,2.');
end
% Append a sub epsilon if noise
if noise_level~=0
  VV=[VV '_\epsilon'];
end

if strcmp(ens,'dct')
  ProbClass=['(DCT,' VV ')'];
elseif strcmp(ens,'smv')
  ProbClass=['({\fontname{zapfchancery}S}_{' num2str(nonzeros) '},' VV ')'];
elseif strcmp(ens,'gen')
  ProbClass=['({\fontname{zapfchancery}N},' VV ')'];
end

km_cell=cell(length(alg_list),1);
error_cell=cell(length(alg_list),1);
time_cell=cell(length(alg_list),1);
km_list = [];
error_list=[];
time_list=[];
  

for j=1:length(alg_list)
  [km_cell{j} error_cell{j} junk] = make_delta_rho_plot(alg_list{j},ensemble,n,succCriteria,nz,noise_level,vecDistr);
  [km_cell{j} time_cell{j} junk] = make_delta_rho_plot(alg_list{j},ensemble,n,'time_for_alg',nz,noise_level,vecDistr);
  km_list = [km_list; km_cell{j}];
  error_list = [error_list; error_cell{j}];
  time_list = [time_list;time_cell{j}];
end


ind=find(error_list<=error_tol);
km_list=km_list(ind,:);
km_list=intersect(km_list,km_list,'rows');
km_list=sortrows(km_list,[2 1]);

best_alg=cell(size(km_list,1),1);
time_ave_list=ones(size(km_list,1),1)*max(time_list); 
error_ave_list=zeros(size(km_list,1),1);

%go through the km_list and see which algorithm has the lowest time 


for j=1:size(km_list,1)
  for i=1:length(alg_list)
    [junk,ind]=intersect(km_cell{i},km_list(j,:),'rows');
    if isempty(ind)==0 %km_list(j,:) was tested for this algorithm
      if ((error_cell{i}(ind)<error_tol) & (time_ave_list(j)>=time_cell{i}(ind)))
      time_ave_list(j)=time_cell{i}(ind);
      error_ave_list(j)=error_cell{i}(ind);
      best_alg{j}=alg_list{i};
      end
    end
  end

  if smoothing
    if j>1 
      if km_list(j,2)==km_list(j-1,2)
        if strcmp(best_alg{j},best_alg{j-1})==0
          if abs(time_ave_list(j)-time_ave_list(j-1))<time_ave_list(j)/20
            best_alg{j}=best_alg{j-1};
          end
        end
      end
    end
  end
  
          
          
  
%  display(sprintf('for k=%d, m=%d, %s is fastest with time=%0.0fms and error=%f',km_list(j,1),km_list(j,2),best_alg{j}(1:end-6),time_ave_list(j),error_ave_list(j)));
%  pause
  
end


%-------------------------------------
% plot a map of which algorithm is best for each kmn


figure(1)

%c_list='krbgmy';
%s_list='o+sxd*';
num_best=zeros(length(alg_list),1);

best_ind_list=zeros(length(km_list),1);

hold off
for j=1:length(km_list)
  best_ind=0;
  for q=1:length(alg_list)
    if strcmp(best_alg{j},alg_list(q))
      best_ind=q;
    end
  end
  best_ind_list(j)=best_ind;
  num_best(best_ind)=num_best(best_ind)+1;
  txt=[c_list(best_ind) s_list(best_ind)];
  plot3(km_list(j,2)/n,km_list(j,1)/km_list(j,2),best_ind,txt)
  hold on
end
hold off
  
view([0 90])
top=0.9;


for i=1:length(alg_list)
  if sum(best_ind_list==i)>minappear
    h=text(0.1,top,0,label_list{i},'fontsize',plotfont);
    set(h,'Color',c_list(i))
    top=top-0.05;
  end
end

axis([0 1 0 1])
xlabel('\delta=m/n','Fontsize',labelfont)
ylabel('\rho=k/m','Fontsize',labelfont,'rotation',90)
title_str = ['Algorithm selection map for ' ProbClass];
if noise_level<=0 
  title_str=sprintf('Algorithm selection map for %s with n = 2^{%d}',ProbClass,log2(n));
else
  title_str=sprintf('Algorithm selection map for %s \\epsilon = %0.1f, n = 2^{%d}',ProbClass,noise_level,log2(n));
end

title(title_str,'Fontsize',titlefont)



fname_out=['plots/algorithm_selection_' ensemble '_n_' num2str(n) vec_str fname_noise_string fig_suffix];
if strcmp('smv',ensemble)
  fname_out=['plots/algorithm_selection_' ensemble '_n_' num2str(n) vec_str fname_noise_string ...
             '_nonzeros_' num2str(nonzeros) fig_suffix];
end
print(print_command,fname_out)
 


algfrac=num_best/sum(num_best);
display(sprintf('Percentage of selection map marked by each algorithm for matrix ensemble %s, vector ensemble %s, noise = %0.2f:',ens,VV, noise_level));
for i=1:length(alg_list)
   display(sprintf('%s \t%0.3f',alg_list{i},algfrac(i)));
end
display('  ');


%----------------------------------------------
% make a plot of the time


figure(2)
%plot the time_ave_list
%exclude the 5% largest values

a=sort(time_ave_list,'descend');
c=round(length(a)*0.05);
thres=a(c);

ind=find(time_ave_list<thres);
time_ave_list=time_ave_list(ind);
km_list=km_list(ind,:);

delta=km_list(:,2)/n;
rho=km_list(:,1)./km_list(:,2);

tri_full=delaunay(delta,rho);
%this makes a triangulation of delta and rho, 
%but some of these triangles go outside the region 
%we are interested in, so we only keep triangles 
%when the vertices in the triangle are nearby

edge_length=0.15;

tri=[];

for j=1:size(tri_full,1)
  d_tri=delta(tri_full(j,:));
  r_tri=rho(tri_full(j,:));
  d_diff=abs([d_tri(2:end)-d_tri(1:end-1); d_tri(1)-d_tri(end)]);
  r_diff=abs([r_tri(2:end)-r_tri(1:end-1); r_tri(1)-r_tri(end)]);
  if max([d_diff])<edge_length
    tri=[tri; tri_full(j,:)];
  end
end

tmp=sort(time_ave_list);
mid=ceil(length(tmp)/2);
mid=tmp(mid);

trisurf(tri,delta,rho,min(time_ave_list,2*mid))
shading interp
view([0 90])
axis([0 1 0 1])
colorbar
xlabel('\delta=m/n','Fontsize',labelfont)
ylabel('\rho=k/m','Fontsize',labelfont,'rotation',90)
if noise_level<=0 
  title_str=sprintf('Time (ms) of fastest algorithm for %s with n = 2^{%d}',ProbClass,log2(n));
else
  title_str=sprintf('Time (ms) of fastest algorithm for %s \\epsilon = %0.1f, n = 2^{%d}',ProbClass,noise_level,log2(n));
end

title(title_str,'Fontsize',titlefont)


fname_out=['plots/best_time_' ensemble '_n_' num2str(n) vec_str fname_noise_string  fig_suffix];
if strcmp('smv',ensemble)
  fname_out=['plots/best_time_' ensemble '_n_' num2str(n) vec_str fname_noise_string ...
             '_nonzeros_' num2str(nonzeros) fig_suffix];
end
print(print_command,fname_out)
 



%-----------------------------------------
% compare the time of the "best" with a fixed algorithm

  
for i=1:length(alg_list)
  km_current = km_cell{i};
  error_current = error_cell{i};
  time_current = time_cell{i};

  figure(2+i)
  hold off;
  ind=find(error_current<=error_tol);
  km_current=km_current(ind,:);
  error_current=error_current(ind);
  time_current=time_current(ind);
  [km_current,junk,ind]=intersect(km_list,km_current,'rows');
  error_current=error_current(ind);
  time_current=time_current(ind);



  time_ratio=zeros(length(km_current),1);
  for j=1:length(time_ratio)
    [junk,ind]=intersect(km_list,km_current(j,:),'rows');
    time_ratio(j)=time_current(j)/time_ave_list(ind);
  end


  delta=km_current(:,2)/n;
  rho=km_current(:,1)./km_current(:,2);

  tri_full=delaunay(delta,rho);

  tri=[];

  for j=1:size(tri_full,1)
    d_tri=delta(tri_full(j,:));
    r_tri=rho(tri_full(j,:));
    d_diff=abs([d_tri(2:end)-d_tri(1:end-1); d_tri(1)-d_tri(end)]);
    r_diff=abs([r_tri(2:end)-r_tri(1:end-1); r_tri(1)-r_tri(end)]);
    if max([d_diff])<edge_length 
      tri=[tri; tri_full(j,:)];
    end
  end

  tmp=sort(time_ratio);
  mid=ceil(length(tmp)/2);
  mid=tmp(mid);

  trisurf(tri,delta,rho,min(10*mid,time_ratio));
  axis([0 1 0 1 0.95 max(time_ratio)]);
  shading interp
  view([0 90])
  axis([0 1 0 1])
  colorbar

  xlabel('\delta=m/n','Fontsize',labelfont)
  ylabel('\rho=k/m','Fontsize',labelfont,'rotation',90)


  if noise_level<=0 
    title_str=sprintf('Time: %s / fastest algorithm for %s with n = 2^{%d}',alg_list{i},ProbClass,log2(n));
  else
    title_str=sprintf('Time: %s / fastest algorithm for %s \\epsilon = %0.1f, n = 2^{%d}',alg_list{i},ProbClass,noise_level,log2(n));
  end

  title(title_str,'Fontsize',titlefont)


  fname_out=['plots/' lower(alg_list{i}) '_ratio_' ensemble '_n_' num2str(n)  vec_str fname_noise_string  fig_suffix];
  if strcmp('smv',ensemble)
    fname_out=['plots/' lower(alg_list{i}) '_ratio_' ensemble '_n_' num2str(n) vec_str fname_noise_string   ...
               '_nonzeros_' num2str(nonzeros) fig_suffix];
  end
  print(print_command,fname_out)
end % ends the alg_list loop for the time ratio plots



