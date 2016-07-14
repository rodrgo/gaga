function make_transition_plots_noise(alg_list, ens_list, noise_list, vecDistr_list,no_zero_noise_str)

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

if nargin<5
  no_zero_noise_str = 1;  % old files often ended in noise0.000
if nargin<4
  vecDistr_list = [1 2 0];
  warning('No vector distribution list specified by user; default [1 2 0] used.')
  if nargin<3
    noise_list = [0 0.1];
    warning('No noise list specified by user; default [0 0.1] used.')
    if nargin<2
      ens_list=cell(1,1);
      ens_list{1}='dct';
      warning('No ensemble list specified by user; default {dct} used.')
      if nargin<1
	alg_list=cell(4,1);
	alg_list{1}='CGIHT';
	alg_list{2}='NIHT';
	alg_list{3}='HTP';
	alg_list{4}='CSMPSP';
	warning('No algorithm list specified by user; default {CGIHT, NIHT, HTP, CSMPSP} used.')
      end
    end
  end
end 
end 
    


axis([0 1 0 1]);

tic
format shortg

[clist slist junk]=set_color_symbol(alg_list);


for pp = 1:length(noise_list)
  noise_level = noise_list(pp);

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

for qq=1:length(vecDistr_list)

vecDistribution = vecDistr_list(qq);

% This defines a string VV for writing the problem class
% as (MM,VV) based on vecDistribution
vec_str = ['_vecDistr' num2str(vecDistribution)];
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
% Append a sub epsilon if there is noise.
eps_str = '';
if noise_level~=0
  VV=[VV '_\epsilon'];
  eps_str = sprintf(' with \\epsilon = %s', num2str(noise_level));
end

for j=1:length(ens_list)
  for i=1:length(alg_list)
    results =[];, betas=[];, deltas=[];
    alg=alg_list{i};
    fname=['results_' alg_list{i} '_S_' ens_list{j} vec_str noise_string '.mat'];
    load(fname)
    
    figure(1)
    hold off

    figure(3)
    hold off
    
    if strcmp(ens_list(j),'dct')
      ind_linfty=7;
      ProbClass=['(DCT,' VV ')'];
    elseif strcmp(ens_list(j),'smv')
      ind_linfty=9;
      %ProbClass=['({\fontname{zapfchancery}S}_{' num2str(nz_list(k)) '},' VV ')'];
    elseif strcmp(ens_list(j),'gen')
      ind_linfty=8;
      ProbClass=['({\fontname{zapfchancery}N},' VV ')'];
    end
    
    if strcmp(ens_list{j},'smv')
      nz_list_full=nz_list;
      nz_list=intersect(nz_list,nz_list);
      for k=1:length(nz_list)
        ind=find(nz_list_full==nz_list(k));
        n=n_list(ind);
	ProbClass=[];
	ProbClass=['({\fontname{zapfchancery}S}_{' num2str(nz_list(k)) '},' VV ')'];
        for zz=1:length(ind)
          figure(1)
          r_star=1./betas{ind(zz)}(:,2);
          r_star10=(1./betas{ind(zz)}(:,2)).*(1+log(1/0.1-1)./betas{ind(zz)}(:,1));
          r_star90=(1./betas{ind(zz)}(:,2)).*(1+log(1/0.9-1)./betas{ind(zz)}(:,1));
          hh=plot(deltas{ind(zz)},r_star,clist(zz));
          if strcmp(alg,'CGIHTprojected')
            set(hh,'color',[1.0,0.6,0]);
          end
          hold on
%          plot(deltas{ind(zz)},r_star10,clist(zz))
%          plot(deltas{ind(zz)},r_star90,clist(zz))
          
          ll=ceil(length(deltas{ind(zz)})*(zz-1/2)/length(ind));
          txt=['\leftarrow n=2^{' num2str(log2(n(zz))) '}'];
          h=text(deltas{ind(zz)}(ll),r_star(ll),txt,'HorizontalAlignment','left','fontsize',plotfont);
          set(h,'Color',clist(zz))
          if strcmp(alg,'CGIHTprojected')
            set(h,'color',[1.0,0.6,0]);
          end
          
          figure(3)
          hh=plot(deltas{ind(zz)},r_star10-r_star90,clist(zz));
          if strcmp(alg,'CGIHTprojected')
            set(hh,'color',[1.0,0.6,0]);
          end
          txt=['\leftarrow n=2^{' num2str(log2(n(zz))) '}'];
          h=text(deltas{ind(zz)}(ll),r_star10(ll)-r_star90(ll),txt,'HorizontalAlignment','left','fontsize',plotfont);
          set(h,'Color',clist(zz))
          if strcmp(alg,'CGIHTprojected')
            set(h,'color',[1.0,0.6,0]);
          end
          hold on
        end		% ends length(ind) loop (zz)
        figure(1)
	title_txt=[ alg_list{i} ' for ' ProbClass eps_str];
        title(title_txt,'fontsize',titlefont)
        xlabel('\delta=m/n','fontsize',labelfont);
        ylabel('\rho=k/m','fontsize',labelfont);
        set(gca,'Fontsize',labelfont);
        fname1=['noiseplots/transition_' alg_list{i} '_' ens_list{j} '_p' num2str(nz_list(k)) vec_str fname_noise_string fig_suffix];
        print(fname1,print_command)
        hold off

        figure(3)
        title_txt=['Gap between 10% and 90% success for ' alg_list{i} ' for ' ProbClass eps_str];
        title(title_txt,'fontsize',titlefont)
        xlabel('\delta=m/n','fontsize',labelfont);
        ylabel('\rho=k/m','fontsize',labelfont);
        set(gca,'Fontsize',labelfont);
        fname2=['noiseplots/transition_width_' alg_list{i} '_' ens_list{j} '_p' num2str(nz_list(k)) vec_str fname_noise_string fig_suffix];
        print(fname2,print_command)
        hold off
%        pause
      end		%ends nz_list loop (k)
    else			 
      for k=1:length(n_list)
        figure(1)
        r_star=1./betas{k}(:,2);
        r_star10=(1./betas{k}(:,2)).*(1+log(1/0.1-1)./betas{k}(:,1));
        r_star90=(1./betas{k}(:,2)).*(1+log(1/0.9-1)./betas{k}(:,1)); 
        hh=plot(deltas{k},r_star,clist(k))
        if strcmp(alg,'CGIHTprojected')
          set(hh,'color',[1.0,0.6,0]);
        end
        hold on
%        plot(deltas{k},r_star10,clist(k))
%        plot(deltas{k},r_star90,clist(k))
        ll=ceil(length(deltas{k})*(k-1/2)/length(n_list));
        txt=['\leftarrow n=2^{' num2str(log2(n_list(k))) '}'];
        h=text(deltas{k}(ll),r_star(ll),txt,'HorizontalAlignment','left','fontsize',plotfont);
        set(h,'Color',clist(k))
        if strcmp(alg,'CGIHTprojected')
          set(h,'color',[1.0,0.6,0]);
        end
        hold on
        
        
        figure(3)
        hh=plot(deltas{k},r_star10-r_star90,clist(k));
        if strcmp(alg,'CGIHTprojected')
          set(hh,'color',[1.0,0.6,0]);
        end
        ll=ceil(length(deltas{k})*(k-1/2)/length(n_list));
        txt=['\leftarrow n=2^{' num2str(log2(n_list(k))) '}'];
        h=text(deltas{k}(ll),r_star10(ll)-r_star90(ll),txt,'HorizontalAlignment','left','fontsize',plotfont);
        set(h,'Color',clist(k))
        if strcmp(alg,'CGIHTprojected')
          set(h,'color',[1.0,0.6,0]);
        end
        hold on
        
      end  % ends n_list loop (k)
      figure(1)
      title_txt=[ alg_list{i} ' for ' ProbClass eps_str];
      title(title_txt,'fontsize',titlefont)
      xlabel('\delta=m/n','fontsize',labelfont);
      ylabel('\rho=k/m','fontsize',labelfont);
      set(gca,'Fontsize',labelfont);
      fname1=['noiseplots/transition_' alg_list{i} '_' ens_list{j} vec_str fname_noise_string fig_suffix];
      print(fname1,print_command)

      figure(3)
      title_txt=['Gap between 10% and 90% success for ' alg_list{i} '  for ' ProbClass eps_str];
      title(title_txt,'fontsize',titlefont)
      xlabel('\delta=m/n','fontsize',labelfont);
      ylabel('\rho=k/m','fontsize',labelfont);
      set(gca,'Fontsize',labelfont);
      fname2=['noiseplots/transition_width_' alg_list{i} '_' ens_list{j} vec_str fname_noise_string fig_suffix];
      print(fname2,print_command)
%      pause
    end  % ends if (ens == smv) conditional
       
    toc
  end  % ends alg_list (j)
end  %ends ens_list  (i) 

end % ends vecDistr_list loop (qq)

end % ends noise_level list loop (pp)


