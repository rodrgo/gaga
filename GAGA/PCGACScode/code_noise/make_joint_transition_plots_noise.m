function make_joint_transition_plots_noise(ens,n,nonzeros,alg,vecDistribution,noise_list,no_zero_noise_str)
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
  noise_list = [0 0.1];
  warning('No noise list specified by user; default [0 0.1] used.')
  if nargin<5
    vecDistribution = 1;
    warning('No vecDistribution specified by user; default 1 used.')
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

vec_str = ['_vecDistr' num2str(vecDistribution)];

clist='brkmgc';


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
VV=[VV '_\epsilon'];

if strcmp(ens,'dct')
  ProbClass=['(DCT,' VV ')'];
elseif strcmp(ens,'smv')
  ProbClass=['({\fontname{zapfchancery}S}_{' num2str(nonzeros) '},' VV ')'];
elseif strcmp(ens,'gen')
  ProbClass=['({\fontname{zapfchancery}N},' VV ')'];
end


figure(1)
hold off

for i=1:length(noise_list)
  noise_level = noise_list(i);
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

  fname=['results_' alg vec_str noise_string '.mat'];
  load(fname)
    
  ind_n=find(n_list==n);
  %size(ind_n)
  if strcmp(ens,'smv')
    ind_nz=find(nz_list==nonzeros);
    ind_n=intersect(ind_n,ind_nz);
  end
  plot(deltas{ind_n},1./betas{ind_n}(:,2),clist(i));
  ll=ceil(length(deltas{ind_n})*(i-1/2)/length(noise_list));
  txt=['\leftarrow \epsilon = ' num2str(noise_level)];
  h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2),txt,'HorizontalAlignment','left','fontsize',plotfont);
  set(h,'color',clist(i))
  hold on
  
end  % ends noise_list loop (i)

%setyaxis=ylim;
%axis([xlim 0 setyaxis(2)]);

axis([0 1 0 1])

txt=['50% phase transition curve for ' alg_short ' for ' ProbClass ...
     ' with n = 2^{' num2str(log2(n)) '}'];
fname1=['noiseplots/transition_allNOISE_' alg];

if strcmp(ens,'smv')
  fname1=[fname1 '_p' num2str(nonzeros)];
end

%{
txt=[txt ', VecDistribution = ' num2str(vecDistribution)];

if strcmp(ens,'smv')
  title(txt,'fontsize',14)
else
  title(txt,'fontsize',14)
end 

%}
%txt = sprintf(txt);
title(txt,'fontsize',titlefont)
fname1=[fname1 vec_str fig_suffix];

xlabel('\delta=m/n','fontsize',labelfont);
ylabel('\rho=k/m','fontsize',labelfont);
print(fname1,print_command)

