function make_joint_algorithms_plots(ens,n,nonzeros,noise_level,vecDistribution,alg_list,no_zero_noise_str)
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
  alg_list=cell(4,1);
  alg_list{1}='ThresholdCG';
  alg_list{2}='NIHT';
  alg_list{3}='HTP';
  alg_list{4}='CSMPSP';
  warning('No algorithm list specified by user; default {ThresholdCG, NIHT, HTP, CSMPSP} used.')
  if nargin<5
    vecDistribution = 1;
    warning('No vecDistribution specified by user; default 1 used.')
    if nargin < 4
      noise_level = 0;
      warning('No noise_level specified by user; default 0 used.')
    end
  end
end
end

if (strcmp(ens,'smv')==1 & nargin<3)
  warning('smv requires nonzeros to be specified')
end


foldername='plots';
if no_zero_noise_str==0
   foldername='noiseplots';
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

vec_str = ''; %['_vecDistr' num2str(vecDistribution)];

%{
if length(alg_list)<4
  clist='rkm';
else 
  clist='brkmgc';
end

% for ISIT CGIHT
clist='krmgb';
%}
% for PCGACS
clist='krb';

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


figure(1)
hold off

for i=1:length(alg_list)
  fname=['results_' alg_list{i} '_S_' ens vec_str noise_string '.mat'];
  load(fname)
    
  ind_n=find(n_list==n);
  if strcmp(ens,'smv')
    ind_nz=find(nz_list==nonzeros);
    ind_n=intersect(ind_n,ind_nz);
  end
  plot(deltas{ind_n},1./betas{ind_n}(:,2),clist(i));
  ll=ceil(length(deltas{ind_n})*(i-1/2)/length(alg_list));
  if i==length(alg_list)
    ll=ceil(length(deltas{ind_n})*(i-1)/length(alg_list));
  end
  if i==1
    ll=ceil(length(deltas{ind_n})*(i)/length(alg_list));
  end
  txt=['\leftarrow ' alg_list{i}];
  h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2),txt,'HorizontalAlignment','left','fontsize',plotfont);
  set(h,'color',clist(i))
  hold on
  


end

%setyaxis=ylim;
%axis([xlim 0 setyaxis(2)]);

axis([0 1 0 1])

if noise_level==0
  txt=sprintf('50%% phase transition curves for %s, n=2^{%d}', ProbClass, log2(n));
else
  txt=sprintf('50%% phase transition curves for %s \\epsilon = %0.1f, n=2^{%d}', ProbClass, noise_level, log2(n));
end

fname1=[foldername '/transition_allALG_' ens];
if strcmp(ens,'smv')
  fname1=[fname1 '_p' num2str(nonzeros)];
end
noise_str=strrep(noise_string,'.','-');
fname1=[fname1 vec_str noise_str fig_suffix];

title(txt,'fontsize',titlefont)


xlabel('\delta=m/n','fontsize',labelfont);
ylabel('\rho=k/m','fontsize',labelfont);
print(fname1,print_command)

