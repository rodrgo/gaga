function [km_list data_ave_list data_sd_list] = make_delta_rho_plot(alg,ens,n,data_type,show_plot,nonzeros)

fname_load=['results_' alg '_S_' ens '.mat'];
load(fname_load);


% we use the data from a regular mesh of delta,rho with data
% generated from the generate_timings scripts.  these mesh of delta
% and rho are as follows.
% delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
% delta_list=transpose(delta_list);
% delta_list=sort(delta_list);
% we extract the delta_list from the logit fits

rho_list=0.02:0.02:0.98;

% the script will make a plot for a specified data type.  the data
% types are: l_one error,l_two error,l_infinity error, time for the
% algorithm, time per iteration, time total, iterations, convergence
% rate, number of true positive, number of false positives, number
% of true negatives, number of false negatives.
% data of these type are contained in the "results" matrix of data,
% column number:

% data type      dct results column number  
% l_one             5    
% l_two             6    
% l_infinity        7
% time for alg      8
% time per iter     9
% time total       10
% iterations       11
% conv. rate       12
% num true pos     13
% num flase pos    14
% num true neg     15
% num false neg    16

% different data types should be presented in log and/or ratios of
% the standard deviation over the median.  the different plot
% options are

% plot_type = 1 indicates log10 of the average and no standard deviation.
%               used for error plots
% plot_type = 2 indicates log10 of the average and log10 of the
%               ratio of the deviation over the average.
%               used for the timing plots
% plot_type = 3 indicates the average and log10 ratio of the standard
%               deviation over the average.  used for support set
%               plots. 



if strcmp(data_type,'l_one')
  ind_results=5;
  plot_type=1;
  title_txt_f1=['Average error (Log10) in the l one norm'];
elseif strcmp(data_type,'l_two')
  ind_results=6;
  plot_type=1;
  title_txt_f1=['Average error (Log10) in the l two norm'];
elseif strcmp(data_type,'l_infinity')
  ind_results=7;
  plot_type=1;
  title_txt_f1=['Average error (Log10) in the l infinity norm'];
elseif strcmp(data_type,'time_for_alg')
  ind_results=8;
  plot_type=2;
  title_txt_f1=['Average time for the algorithm (Log10)'];
  title_txt_f2=['(Log10) Ratio of average time deviation for the algorithm over the average'];
elseif strcmp(data_type,'time_per_iter')
  ind_results=9;
  plot_type=2;
  title_txt_f1=['Average time per iteration (Log10)'];
  title_txt_f2=['(Log10) Ratio of average time deviation per iteration over the average'];
elseif strcmp(data_type,'time_total')
  ind_results=10;
  plot_type=2;
  title_txt_f1=['Average total time (Log10)'];
  title_txt_f2=['(Log10) Ratio of average total time deviation over the average'];
elseif strcmp(data_type,'iterations')
  ind_results=11;
  plot_type = 2;
  title_txt_f1=['Average number of iterations (Log10)'];
  title_txt_f2=['(Log10) Ratio of average number of iterations deviation over the average'];
elseif strcmp(data_type,'conv_rate')
  ind_results=12;
  plot_type = 2;
  title_txt_f1=['Average convergence rate (Log10)'];
  title_txt_f2=['(Log10) Ratio of average convergence rate deviation over the average'];
elseif strcmp(data_type,'true_pos')
  ind_results=13;
  plot_type = 3;
  title_txt_f1=['Average fraction of support set correct'];
  title_txt_f2=['(Log10) Ratio of average fractional support set correct deviation over the average'];
elseif strcmp(data_type,'false_pos')
  ind_results=14;
  display('plotting format not set')
elseif strcmp(data_type,'true_neg')
  ind_results=15;
  display('plotting format not set')
elseif strcmp(data_type,'false_neg')
  ind_results=16;
  display('plotting format not set')
else
  error('data_type not supported, check script.')
end

% smv has an offset of 2 in the column number, so iterations is in
% column 13 rather than 11.
% gen has an offset of 1 in the column number, so iterations is in
% column 12 rather than 11.

ens_offset=0;
if strcmp(ens,'dct')
  ens_offset=0;
elseif strcmp(ens,'smv')
  ens_offset=2;
elseif strcmp(ens,'gen')
  ens_offset=1;
end

%correct the offset
ind_results=ind_results+ens_offset;


km_list=[];
data_ave_list=[];
data_sd_list=[];

% select the values of n (and possibly nonzeros) specified, in
% ind_n and extract the correct logit fit parameters in ind_tmp
if strcmp(ens,'smv')
  ind_n=find(results(:,3)==n & results(:,5)==nonzeros);
  ind_tmp=find(n_list==n & nz_list==nonzeros);
else
  ind_n=find(results(:,3)==n);
  ind_tmp=find(n_list==n);
end



logit_betas=betas{ind_tmp}(:,:);
logit_deltas=deltas{ind_tmp}(:);


delta_list=deltas{ind_tmp}(:);
m_list=ceil(n*delta_list);
m_list=sort(m_list);

% the 50% success rate
r_star0=1./logit_betas(:,2);
k_star0=round(n*logit_deltas.*r_star0);

% the 10% success rate
%r_star=(1./logit_betas(:,2)).*(1+log(9)./logit_betas(:,1));
%k_star=round(n*logit_deltas.*r_star);

% display those k that are no more than 10% above the 50% success curve.
k_star=1.1*k_star0;
% unless data_typ=='true_pos' then use all data
if strcmp(data_type,'true_pos')
  k_star=m_list;
end


%[k_star0 k_star]

if strcmp(data_type,'conv_rate')
  % for conv_rate only use the data with at least three iterations.
  ind_tmp=find(results(:,ind_results-1)>1);
  ind_n=intersect(ind_n,ind_tmp);
end

% extract the data to a three column matrix, with those columns
% being: k, m, and the values of the data type selected
results=results(ind_n,[1 2 ind_results]);


if strcmp(data_type,'iterations')
  results(:,3)=results(:,3)+1; %counter is off by one
elseif strcmp(data_type,'true_pos')
  results(:,3)=results(:,3)./results(:,1); %fraction of supp set correct
end
results(:,3)=max(results(:,3),10^(-5)); %for log in plots


for ll=1:length(m_list)
  m=m_list(ll);
    
  k_list=ceil(m*rho_list);
  k_list=max(k_list,1); k_list=min(k_list,m-1);
  k_list=intersect(k_list,k_list);
  k_list_gap=max(k_list(2:end)-k_list(1:end-1));
  if length(k_list)<2 % in case k_list has only one entry
    k_list_gap=0;
  end
  
  
  for qq=1:length(k_list)
    k=k_list(qq);
    
    % extract the data that is sufficiently close to k, based upon:
    if k_list_gap <= 2
      k_range=0;
    else
      k_range=ceil(0.1*k_list_gap);
    end
    

    ind=find( abs(k-results(:,1))<=k_range & results(:,2)==m);
    
%    results(ind,1:3)
%    length(ind)
%    [k m]
%    k/m
%    pause
    
    if (prod(size(ind))>0 & k<k_star(ll))
      km_list=[km_list; [k m]];
      
      tmp_data=results(ind,3);
      data_ave_list=[data_ave_list; sum(tmp_data)/length(tmp_data)];
      data_sd_list=[data_sd_list; norm(tmp_data-data_ave_list(end),2)/sqrt(length(ind))];
      
    end
  end
end



%sd_over_ave=data_sd_list./data_ave_list;
%ind=find(sd_over_ave>1);
%[km_list(ind,1) km_list(ind,2) sd_over_ave(ind)]


if prod(km_list)>0
  rho=km_list(:,1)./km_list(:,2);
  delta=km_list(:,2)/n;

  [delta_mesh,rho_mesh]=meshgrid(delta_list,rho_list);

  data_ave_mesh=griddata(delta,rho,data_ave_list,delta_mesh,rho_mesh);
  data_sd_mesh=griddata(delta,rho,data_sd_list,delta_mesh,rho_mesh);

  %remove any zeros so that logs can be taken for contour plots
  ind_sd=find(data_sd_list>0);
  min_sd=min(data_sd_list(ind_sd));
  ind_ave=find(data_ave_list>0);
  min_ave=min(data_ave_list(ind_sd));
  for i=1:size(data_sd_mesh,1)
    for j=1:size(data_sd_mesh,2)
      if isfinite(data_sd_mesh(i,j))==0
        data_sd_mesh(i,j)=min_sd;
      end
      if isfinite(data_ave_mesh(i,j))==0
        data_sd_mesh(i,j)=min_ave;
      end
    end
  end
  data_sd_mesh=max(data_sd_mesh,min_sd);
  
  fname1=['plots/delta_rho_' alg '_' ens '_' data_type];
  fname2=[fname1 '_sd_'];
  fname1=[fname1 '_n' num2str(n)];
  fname2=[fname2 '_n' num2str(n)];
  if strcmp(ens,'smv')
    fname1=[fname1 '_nz' num2str(nonzeros)];
    fname2=[fname2 '_nz' num2str(nonzeros)];
  end
  
  display(fname1)
  
  if show_plot==1
    if plot_type==1
      figure(1)
      hold off
      contour(delta_list,rho_list,log10(data_ave_mesh),20);
      title(title_txt_f1,'Fontsize',14);
      colorbar
      xlabel('delta')
      ylabel('rho')
      print(fname1,'-dpdf')
    elseif plot_type==2
      figure(1)
      hold off
      contour(delta_list,rho_list,log10(data_ave_mesh),20);
      title(title_txt_f1,'Fontsize',14);
      colorbar
      xlabel('delta')
      ylabel('rho')
      print(fname1,'-dpdf')
      figure(2)
      hold off
      contour(delta_list,rho_list,log10(data_sd_mesh./data_ave_mesh),20);
      title(title_txt_f2,'Fontsize',14);
      colorbar
      xlabel('delta')
      ylabel('rho')
      print(fname2,'-dpdf')
    elseif plot_type==3
      figure(1)
      hold off
      contour(delta_list,rho_list,data_ave_mesh,20);
      title(title_txt_f1,'Fontsize',14);
      colorbar
      xlabel('delta')
      ylabel('rho')
      print(fname1,'-dpdf')
      figure(2)
      hold off
      contour(delta_list,rho_list,log10(data_sd_mesh./data_ave_mesh),20);
      title(title_txt_f2,'Fontsize',14);
      colorbar
      xlabel('delta')
      ylabel('rho')
      print(fname2,'-dpdf')
    end
  end
  
  
end

