num_tests = 25;
tol = 10^(-6);

stime=tic;

sd_recovery = 0;
cg_recovery = 0;
cgr_recovery = 0;

cgr_restart_flag = 1; % when equal to one we resort to NIHT,
                      % otherwise use CGIHT on a restricted column space.


show_plots = 0;
r_list = 10:10:40;
  niht_errors=zeros(length(r_list),num_tests);
  cgiht_errors=niht_errors;
  niht_times = niht_errors;
  cgiht_times = niht_errors;

for rr=1:length(r_list)
 r=r_list(rr);
for jj = 1 : num_tests
  m = 1000;
  n = 1000;
  
  %delta = 0.25;
  %rho = 0.5;
  
  %p = round(delta*m*n)
  %r = round(rho*p/(m+n))
  p = 250000;
  %r = 80;

  options=cell(4,2);
  options{1,1}='maxiter';
  options{1,2}=int32(2000);
  options{2,1}='tol';
  options{2,2} = single(10^(-6));
  options{3,1}='PSVDtol';
  options{3,2}=single(.00001);
  options{4,1}='PSVDmaxiter';
  options{4,2}=int32(3);

  [gpu_norms gpu_times gpu_iter gpu_conv gpu_MatOut gpu_MatInput gpu_A] = gagamc_entry('CGIHT',int32(m), int32(n), int32(r), int32(p),options); 
cgiht_errors(rr,jj)=gpu_norms(1); cgiht_times(rr,jj)=gpu_times(2);, cgiht_iter(rr,jj)=gpu_iter;


  [gpu_norms gpu_times gpu_iter gpu_conv gpu_MatOut gpu_MatInput gpu_A] = gagamc_entry('NIHT',int32(m), int32(n), int32(r), int32(p),options); 
niht_errors(rr,jj)=gpu_norms(1); niht_times(rr,jj)=gpu_times(2);, niht_iter(rr,jj)=gpu_iter;

end % end for jj=1:num_tests
display(sprintf('Finished rank = %d',r))
 end % end for rr=1:length(r_list)


niht_ave_err = sum(niht_errors,2)/num_tests;
cgiht_ave_err = sum(cgiht_errors,2)/num_tests;

niht_ave_times = sum(niht_times,2)/num_tests;
cgiht_ave_times = sum(cgiht_times,2)/num_tests;

% convert to seconds
niht_ave_times = niht_ave_times/1000;
cgiht_ave_times = cgiht_ave_times/1000;

niht_ave_iter = sum(niht_iter,2)/num_tests;
cgiht_ave_iter = sum(cgiht_iter,2)/num_tests;



for rr=1:length(r_list)
display(sprintf('Average errors, rank=%d\n   NIHT:  %f\n    CGIHT:  %f\n \n Average times (seconds),rank=%d\n     NIHT:  %f\n    CGIHT:  %f\n \n Average iterations,rank=%d\n     NIHT:  %0.2f\n    CGIHT:  %0.2f\n \n',r_list(rr),niht_ave_err(rr),cgiht_ave_err(rr),r_list(rr),niht_ave_times(rr),cgiht_ave_times(rr),r_list(rr),niht_ave_iter(rr),cgiht_ave_iter(rr)))  
end

total_time = toc(stime)
