function accel_factors = calculate_cpu_gpu_acceleration(alg,matrixEnsemble,n_select,nonZeros_select)

show_plots=0;  %show_plot=1 shows the plots, and 0 does not


% alg will be either: NIHT or HTP
% matrixEnsemble will be either: dct, smv, or gen

% n_select is the number of columns

% nonZeros_select is only for matrixEnsemble smv and is the number of
% nonzeros per column 

% makes a plot of the time for cpu vs gpu


% the other variants are always one choice in the data considered here
vecDistribution=1; %0 is for uniform [0,1], 
                   %1 is for random {-1,1},
                   %2 is for gaussian.


% supp_flag is: 0 for dynamic binning, 1 is for always binning,
%               2 for sort with dynamics, and 3 is for always sorting
supp_flag=0; 

fname_gpu=sprintf('results_%s_%s_timing_supp_flag_%d',alg,matrixEnsemble,supp_flag);

fname_cpu=sprintf('results_%s_%s_matlab_timing_supp_flag_%d',alg,matrixEnsemble,supp_flag);


m_minimum=1;  %can be used to remove data for unrealistically small
              %problem sizes.  using medians makes this less necessary


load(fname_gpu)
%if matrixEnsemble='smv' then need to extract the data for
%nonZeros_select and then rename the data so that it is the same
%for all ensembles
if strcmp(matrixEnsemble,'smv')
  ind=find(kmnp_list(:,4)==nonZeros_select);
  kmn_list=kmnp_list(ind,1:3);
  results=results(ind);
  time_per_iteration_kmn_all=time_per_iteration_kmnp_all(ind);
  time_supp_set_kmn_all=time_supp_set_kmnp_all(ind);
  clear kmnp_list time_per_iteration_kmnp_all time_supp_set_kmnp_all;
  if strcmp(alg,'HTP')
    time_per_cg_iteration_kmn_all=time_per_cg_iteration_kmnp_all(ind);
    total_cg_steps_kmn_all=total_cg_steps_kmnp_all(ind);
    clear time_per_cg_iteration_kmnp_all total_cg_steps_kmnp_all;
  end
end


%extract the data for n_select and rename the gpu results with _gpu at the end
ind=find(kmn_list(:,3)==n_select & kmn_list(:,2)>=m_minimum);
kmn_list_gpu=kmn_list(ind,:);
results_gpu=results(ind);
time_per_iteration_kmn_all_gpu=time_per_iteration_kmn_all(ind);
time_supp_set_kmn_all_gpu=time_supp_set_kmn_all(ind);
if strcmp(alg,'HTP')
  time_per_cg_iteration_kmn_all_gpu=time_per_cg_iteration_kmn_all(ind);
  total_cg_steps_kmn_all_gpu=total_cg_steps_kmn_all(ind);
end




load(fname_cpu)
%if matrixEnsemble='smv' then need to extract the data for
%nonZeros_select and then rename the data so that it is the same
%for all ensembles
if strcmp(matrixEnsemble,'smv')
  ind=find(kmnp_list(:,4)==nonZeros_select);
  kmn_list=kmnp_list(ind,1:3);
  results=results(ind);
  time_per_iteration_kmn_all=time_per_iteration_kmnp_all(ind);
  time_supp_set_kmn_all=time_supp_set_kmnp_all(ind);
  clear kmnp_list time_per_iteration_kmnp_all time_supp_set_kmnp_all;
  if strcmp(alg,'HTP')
    time_per_cg_iteration_kmn_all=time_per_cg_iteration_kmnp_all(ind);
    total_cg_steps_kmn_all=total_cg_steps_kmnp_all(ind);
    clear time_per_cg_iteration_kmnp_all total_cg_steps_kmnp_all;
  end
end

%extract the data for n_select rename the cpu results with _cpu at the end
ind = find(kmn_list(:,3)==n_select);
kmn_list_cpu=kmn_list(ind,:);
results_cpu=results(ind);
time_per_iteration_kmn_all_cpu=time_per_iteration_kmn_all(ind);
time_supp_set_kmn_all_cpu=time_supp_set_kmn_all(ind);
if strcmp(alg,'HTP')
  time_per_cg_iteration_kmn_all_cpu=time_per_cg_iteration_kmn_all(ind);
  total_cg_steps_kmn_all_cpu=total_cg_steps_kmn_all(ind);
  clear time_per_cg_iteration_kmn_all total_cg_steps_kmn_all;
end


clear time_per_iteration_kmn_all time_supp_set_kmn_all;

%find the values of kmn_list_cpu and kmn_list_gpu that both have data
[kmn_list,ind_gpu,ind_cpu]=intersect(kmn_list_gpu,kmn_list_cpu,'rows');


results_gpu=results_gpu(ind_gpu);
time_per_iteration_kmn_all_gpu=time_per_iteration_kmn_all_gpu(ind_gpu);
time_supp_set_kmn_all_gpu=time_supp_set_kmn_all_gpu(ind_gpu);
results_cpu=results_cpu(ind_cpu);
time_per_iteration_kmn_all_cpu=time_per_iteration_kmn_all_cpu(ind_cpu);
time_supp_set_kmn_all_cpu=time_supp_set_kmn_all_cpu(ind_cpu);
if strcmp(alg,'HTP')
  time_per_cg_iteration_kmn_all_gpu=time_per_cg_iteration_kmn_all_gpu(ind_gpu);
  total_cg_steps_kmn_all_gpu=total_cg_steps_kmn_all_gpu(ind_gpu);
  time_per_cg_iteration_kmn_all_cpu=time_per_cg_iteration_kmn_all_cpu(ind_cpu);
  total_cg_steps_kmn_all_cpu=total_cg_steps_kmn_all_cpu(ind_cpu);
end


clear ind_cpu ind_gpu kmn_list_cpu kmn_list_gpu ind

accel_ratio_generation=zeros(size(kmn_list,1),1);
accel_ratio_descent=zeros(size(kmn_list,1),1);
accel_ratio_supp=zeros(size(kmn_list,1),1);
if strcmp(alg,'HTP')
  accel_ratio_cg=zeros(size(kmn_list,1),1);
end


for j=1:size(kmn_list,1)
  
  mid_gpu=ceil(length(results_gpu{j}(:,1))/2);
  mid_cpu=ceil(length(results_cpu{j}(:,1))/2);
  a=sort(results_gpu{j}(:,3)-results_gpu{j}(:,2));
  a=a(mid_gpu);
  b=sort(results_cpu{j}(:,3)-results_cpu{j}(:,2));
  b=b(mid_cpu);
  accel_ratio_generation(j)=b/a;
    
  mid_gpu=ceil(length(time_per_iteration_kmn_all_gpu{j})/2);
  mid_cpu=ceil(length(time_per_iteration_kmn_all_cpu{j})/2);
  accel_ratio_descent(j)=time_per_iteration_kmn_all_cpu{j}(mid_cpu)/time_per_iteration_kmn_all_gpu{j}(mid_gpu);

  mid_gpu=ceil(length(time_supp_set_kmn_all_gpu{j})/2);
  mid_cpu=ceil(length(time_supp_set_kmn_all_cpu{j})/2);
  accel_ratio_supp(j)=time_supp_set_kmn_all_cpu{j}(mid_cpu)/time_supp_set_kmn_all_gpu{j}(mid_gpu);

  
  if strcmp(alg,'HTP')
    mid_gpu=ceil(length(time_per_cg_iteration_kmn_all_gpu{j})/2);
    mid_cpu=ceil(length(time_per_cg_iteration_kmn_all_cpu{j})/2);
    if mid_gpu*mid_cpu>0
      accel_ratio_cg(j)=time_per_cg_iteration_kmn_all_cpu{j}(mid_cpu)/time_per_cg_iteration_kmn_all_gpu{j}(mid_gpu);
    end
  end

end

%accel_ratio_descent-accel_ratio_supp

if show_plots==1

  hold off
  plot(sort(accel_ratio_descent,'descend'),'k')
  hold on
  plot(sort(accel_ratio_supp,'descend'),'r')
  plot(sort(accel_ratio_generation,'descend'),'m')
  if strcmp(alg,'HTP')
    plot(sort(accel_ratio_cg,'descend'),'b')
  end
  hold off
end


%calculate the median acceleration factors
accel_factors = [];

accel_factors = [accel_factors sum(accel_ratio_descent)/length(accel_ratio_descent)];
accel_factors = [accel_factors sum(accel_ratio_supp)/length(accel_ratio_supp)];
accel_factors = [accel_factors sum(accel_ratio_generation)/length(accel_ratio_generation)];


if strcmp(alg,'HTP')
  mid=ceil(length(accel_ratio_cg)/2);
  median=sort(accel_ratio_cg,'descend');
  median=median(mid);
  accel_factors = [accel_factors median];
end


return









a=sprintf('The data for %s with %s and n = %d',alg,matrixEnsemble,n_select);
if strcmp(matrixEnsemble,'smv')
  a=[a sprintf(' and nonZeros = %d',nonZeros_select)];
end
a=[a sprintf(' is:')];
display(sprintf('%s',a))

ave=sum(accel_ratio_descent)/length(accel_ratio_descent);
mid=ceil(length(accel_ratio_descent)/2);
median=sort(accel_ratio_descent,'descend');
median=median(mid);
display(sprintf('descent acceleration: average %f and median %f',ave,median))

ave=sum(accel_ratio_supp)/length(accel_ratio_supp);
mid=ceil(length(accel_ratio_supp)/2);
median=sort(accel_ratio_supp,'descend');
median=median(mid);
display(sprintf('support acceleration: average %f and median %f',ave,median))

if strcmp(alg,'HTP')
  ave=sum(accel_ratio_cg)/length(accel_ratio_cg);
  mid=ceil(length(accel_ratio_cg)/2);
  median=sort(accel_ratio_cg,'descend');
  median=median(mid);
  display(sprintf('CG acceleration: average %f and median %f',ave,median))
end

display(sprintf('\n'))

pause


