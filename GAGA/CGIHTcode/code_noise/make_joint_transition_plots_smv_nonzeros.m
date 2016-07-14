function make_joint_transition_plots_smv_nonzeros(n,nz_list,alg,noise_level,vecDistribution,no_zero_noise_str)


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

% set font sizes in the plots
% labelfont is for the axis labels, plotfont for text in the plot, titlefont for the title
labelfont = 16;
plotfont = 16;
titlefont = 16;

if nargin<7
  no_zero_noise_str = 1;  % old files often ended in noise0.000
if nargin<5
  vecDistr_list = [1 2 0];
  warning('No vector distribution list specified by user; default [1 2 0] used.')
  if nargin<4
    noise_level = 0;
    warning('No noise level specified by user; default 0 used.')
    if nargin < 3
      alg = 'NIHT';
      warning('No algorithm specified by user; default NIHT used.')
    end
  end
end
end

ens = 'smv';
alg_short=alg;
alg = [alg '_S_' ens];


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

[clist slist llist]=set_color_symbol(alg_list);
%clist='brkmgc';


% This defines a string VV for writing the problem class
% as (MM,VV) based on vecDistribution
if vecDistribution == 0
  VV='U';
elseif vecDistribution == 1
  VV='B';
elseif vecDistribution == 2
  VV='N';
else
  VV='VV';
  warning('vecDistribution was not from the list 0,1,2.');
end
% Append a sub epsilon 
if noise_level~=0
  VV=[VV '_\epsilon'];
end

ProbClass=['({\fontname{zapfchancery}S}_p,' VV ')'];



figure(1)
hold off

for i=1:length(nz_list)
  nonzeros = nz_list(i);
  vec_str = ['_vecDistr' num2str(vecDistribution)];

  fname=['results_' alg vec_str noise_string '.mat'];
  load(fname)
    
  ind_n=find(n_list==n);
  ind_nz=find(nz_list==nonzeros);
  ind_n=intersect(ind_n,ind_nz);
   
  hh=plot(deltas{ind_n},1./betas{ind_n}(:,2),clist{i});
  if strcmp(alg_short,'CGIHTprojected')
    set(hh,'color',[1.0,0.6,0]);
  end
  ll=ceil(length(deltas{ind_n})*(i-1/2)/length(nz_list));
  if i==length(nz_list)
    ll=ceil(length(deltas{ind_n})*(i-1)/length(nz_list));
  end
  if i==1
    ll=ceil(length(deltas{ind_n})*(i)/length(nz_list));
  end
  txt=['\leftarrow p = ' num2str(nonzeros)];
  h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2),txt,'HorizontalAlignment','left','fontsize',plotfont);
  set(h,'color',clist{i})
  if strcmp(alg_short,'CGIHTprojected')
    set(h,'color',[1.0,0.6,0]);
  end
  hold on
  
end  % ends nz_list loop (i)


axis([0 1 0 1])

if noise_level<=0
  txt=sprintf('50%% phase transition curves for %s for %s, n=2^{%d}', alg_short, ProbClass, log2(n));
else
  txt=sprintf('50%% phase transition curves for %s for %s \\epsilon = %0.1f, n=2^{%d}', alg_short, ProbClass, noise_level, log2(n));
end

fname1=['noiseplots/transition_allNZ_' alg];

noise_str=strrep(noise_string,'.','-');
fname1=[fname1 vec_str noise_str fig_suffix];

title(txt,'fontsize',titlefont)

xlabel('\delta=m/n','fontsize',labelfont);
ylabel('\rho=k/m','fontsize',labelfont);
print(fname1,print_command)

