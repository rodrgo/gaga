function make_joint_transition_plots_distribution(ens,n,nonzeros,alg,noise_level,vecDistr_list,no_zero_noise_str)
% ens should be 'dct', 'smv', or 'gen'
% for ens=='smv' the third (nonzeros) argument is required

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
if nargin<6
  vecDistr_list = [1 2 0];
  warning('No vector distribution list specified by user; default [1 2 0] used.')
  if nargin<5
    noise_level = 0;
    warning('No noise level specified by user; default 0 used.')
    if nargin < 4
      alg = 'NIHT';
      warning('No algorithm specified by user; default NIHT used.')
    end
  end
end
end

if (strcmp(ens,'smv')==1 & nargin<3)
  warning('smv requires nonzeros to be specified')
end
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

clist='brkmgc';


% This defines a string VV for writing the problem class
% as (MM,VV) based on vecDistribution


if strcmp(ens,'dct')
  ProbClass=['(DCT,vec)'];
elseif strcmp(ens,'smv')
  ProbClass=['({\fontname{zapfchancery}S}_{' num2str(nonzeros) '},vec)'];
elseif strcmp(ens,'gen')
  ProbClass=['({\fontname{zapfchancery}N},vec)'];
end



figure(1)
hold off

for i=1:length(vecDistr_list)
  vecDistribution = vecDistr_list(i);
  vec_str = ['_vecDistr' num2str(vecDistribution)];

  switch vecDistribution 
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
  if noise_level~=0   
    VV=[VV '_\epsilon'];
  end

  fname=['results_' alg vec_str noise_string '.mat'];
  load(fname)
    
  ind_n=find(n_list==n);
  if strcmp(ens,'smv')
    ind_nz=find(nz_list==nonzeros);
    ind_n=intersect(ind_n,ind_nz);
  end
  plot(deltas{ind_n},1./betas{ind_n}(:,2),clist(i));
  ll=ceil(length(deltas{ind_n})*(i-1/2)/length(vecDistr_list));
  if i==length(vecDistr_list)
    ll=ceil(length(deltas{ind_n})*(i-1)/length(vecDistr_list));
  end
  if i==1
    ll=ceil(length(deltas{ind_n})*(i)/length(vecDistr_list));
  end
  txt=['\leftarrow vec = ' VV];
  if noise_level~=0
    h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2)-.02,txt,'HorizontalAlignment','left','fontsize',plotfont);
  else
    h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2),txt,'HorizontalAlignment','left','fontsize',plotfont);
  end
  set(h,'color',clist(i))
  hold on
  
end  % ends vecDistr_list loop (i)

axis([0 1 0 1])

if noise_level<=0
  txt=sprintf('50%% phase transition curves for %s for %s, n=2^{%d}', alg_short, ProbClass, log2(n));
else
  txt=sprintf('50%% phase transition curves for %s for %s \\epsilon = %0.1f, n=2^{%d}', alg_short, ProbClass, noise_level, log2(n));
end

fname1=['noiseplots/transition_allVECDISTR_' alg];
if strcmp(ens,'smv')
  fname1=[fname1 '_p' num2str(nonzeros)];
end
noise_str=strrep(noise_string,'.','-');
fname1=[fname1 noise_str fig_suffix];

title(txt,'fontsize',titlefont)

xlabel('\delta=m/n','fontsize',labelfont);
ylabel('\rho=k/m','fontsize',labelfont);
print(fname1,print_command)

