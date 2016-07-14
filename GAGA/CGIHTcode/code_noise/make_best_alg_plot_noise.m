function [km_list best_alg time_ave_list error_ave_list] = make_best_alg_plot_noise(ensemble,n,nonzeros,noise_level,vecDistr,alg_list,no_zero_noise_str)

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

% alg_list must contain at least 2 and no more than 6 algorithms

if nargin<7
  no_zero_noise_str = 1;  % old files often ended in noise0.000
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
end

% label_list, c_list, s_list should be defined to label algorithms in selection plots.
% For consistency across papers, the color, symbol, and label are generated by the funciton set_color_symbol
[c_list s_list label_list]=set_color_symbol(alg_list);

% minappear is the minimum number of times a symbol must appear
% in the selection map in order to be labeled
minappear = 1;

% to use l_infity criteria, set l_inf_success = 1
% to use l_two criteria (for noise), set l_inf_success = 0
l_inf_success = 0;
if l_inf_success
  succCriteria = 'l_infinity'
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


if length(alg_list)<2
  error('alg_list must contain at least 2 algorithms.');
end

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

vec_str = ['_vecDistr' num2str(vecDistr)];


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
  
[km_cell{1} error_cell{1} junk] = make_delta_rho_plot_noise(alg_list{1},ensemble,n,succCriteria,nz,noise_level,vecDistr);
[km_cell{2} error_cell{2} junk] = make_delta_rho_plot_noise(alg_list{2},ensemble,n,succCriteria,nz,noise_level,vecDistr);

[km_cell{1} time_cell{1} junk] = make_delta_rho_plot_noise(alg_list{1},ensemble,n,'time_for_alg',nz,noise_level,vecDistr);
[km_cell{2} time_cell{2} junk] = make_delta_rho_plot_noise(alg_list{2},ensemble,n,'time_for_alg',nz,noise_level,vecDistr);

km_list = [km_cell{1}; km_cell{2}];
error_list = [error_cell{1}; error_cell{2}];
time_list = [time_cell{1}; time_cell{2}];

if length(alg_list)>2
 for jj=3:length(alg_list)
  [km_cell{jj} error_cell{jj} junk] = make_delta_rho_plot_noise(alg_list{jj},ensemble,n,succCriteria,nz,noise_level,vecDistr);
  [km_cell{jj} time_cell{jj} junk] = make_delta_rho_plot_noise(alg_list{jj},ensemble,n,'time_for_alg',nz,noise_level,vecDistr);
  km_list = [km_list; km_cell{jj}];
  error_list = [error_list; error_cell{jj}];
  time_list = [time_list;time_cell{jj}];
 end
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
  for ii=1:length(alg_list)
    km_list_active = km_cell{ii};
    error_list_active = error_cell{ii};
    time_list_active = time_cell{ii};
    [junk,ind]=intersect(km_list_active,km_list(j,:),'rows');
    if isempty(ind)==0 %km_list(j,:) was tested for this algorithm
      if ((error_list_active(ind)<error_tol) & (time_ave_list(j)>=time_list_active(ind)))
        time_ave_list(j)=time_list_active(ind);
        error_ave_list(j)=error_list_active(ind);
        best_alg{j}=alg_list{ii};
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
  txt=[c_list{best_ind} s_list{best_ind}];
  hh=plot3(km_list(j,2)/n,km_list(j,1)/km_list(j,2),best_ind,txt);
  if strcmp(alg_list{best_ind},'CGIHTprojected')
    set(hh,'Color',[1.0 0.6 0]);
  end
  hold on
end
hold off
  
view([0 90])
top=0.9;


for i=1:length(alg_list)
  if sum(best_ind_list==i)>minappear
    h=text(0.1,top,0,label_list{i},'fontsize',plotfont);
    set(h,'Color',c_list{i})
    if strcmp(alg_list{i},'CGIHTprojected')
      set(h,'Color',[1, 0.6, 0]);
    end
    top=top-0.05;
  end
end

axis([0 1 0 1])
xlabel('\delta=m/n','Fontsize',labelfont)
ylabel('\rho=k/m','Fontsize',labelfont,'rotation',90)
title_str = ['Algorithm selection map for ' ProbClass];
if noise_level<=0 
  title_str=sprintf('Algorithm selection map for %s, n = 2^{%d}',ProbClass,log2(n));
else
  title_str=sprintf('Algorithm selection map for %s \\epsilon = %0.1f, n = 2^{%d}',ProbClass,noise_level,log2(n));
end

title(title_str,'Fontsize',titlefont)



fname_out=['noiseplots/algorithm_selection_' ensemble '_n_' num2str(n) vec_str fname_noise_string fig_suffix];
if strcmp('smv',ensemble)
  fname_out=['noiseplots/algorithm_selection_' ensemble '_n_' num2str(n) vec_str fname_noise_string ...
             '_nonzeros_' num2str(nonzeros) '.pdf'];
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
hold off

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
  title_str=sprintf('Time (ms) of fastest algorithm for %s, n = 2^{%d}',ProbClass,log2(n));
else
  title_str=sprintf('Time (ms) of fastest algorithm for %s \\epsilon = %0.1f, n = 2^{%d}',ProbClass,noise_level,log2(n));
end

title(title_str,'Fontsize',titlefont)


fname_out=['noiseplots/best_time_' ensemble '_n_' num2str(n) vec_str fname_noise_string  fig_suffix];
if strcmp('smv',ensemble)
  fname_out=['noiseplots/best_time_' ensemble '_n_' num2str(n) vec_str fname_noise_string ...
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
    title_str=sprintf('Time ratio: %s / fastest algorithm for %s, n = 2^{%d}',alg_list{i},ProbClass,log2(n));
  else
    title_str=sprintf('Time ratio: %s / fastest algorithm for %s \\epsilon = %0.1f, n = 2^{%d}',alg_list{i},ProbClass,noise_level,log2(n));
  end

  title(title_str,'Fontsize',titlefont)


  fname_out=['noiseplots/' lower(alg_list{i}) '_ratio_' ensemble '_n_' num2str(n)  vec_str fname_noise_string  fig_suffix];
  if strcmp('smv',ensemble)
    fname_out=['noiseplots/' lower(alg_list{i}) '_ratio_' ensemble '_n_' num2str(n) vec_str fname_noise_string   ...
               '_nonzeros_' num2str(nonzeros) fig_suffix];
  end
  print(print_command,fname_out)
end % ends the alg_list loop for the time ratio plots

