%Chenshuai Sui 2014

num_tests = 1;
tol = 10^(-5);

stime=tic;

cgr_restart_flag = 1; % when equal to one we resort to NIHT,
                      % otherwise use CGIHT on a restricted column space.

alg_list=cell(4,1);
alg_list{1}='LMaFit';
alg_list{4}='SVT';
alg_list{2}='NIHT';
alg_list{3}='CGIHT';



r_list = 10;

%for rr=1:length(r_list)
 % r=r_list(rr);
  niht_errors=zeros(length(r_list),num_tests);
  cgiht_errors=niht_errors;
  niht_times = niht_errors;
  cgiht_times = niht_errors;
  matlab_errors = niht_errors;
  matlab_times = niht_errors;

%for rr=1:length(r_list)
% r=r_list(rr);



for jj = 1 : num_tests
  m = 100;
  n = 100;
  
  %delta = 0.7;
  %rho = 0.2;
  
  %p = round(delta*m*n)
  %r = round(((m+n)-sqrt((m+n)^2-4*p*rho))/2)
  
  r = 3
  p = 8000

  myrho = r*(m+n-r)/p;

      
   

  options=cell(4,2);
  options{1,1}='maxiter';
  options{1,2}=int32(5000);
  options{2,1}='tol';
  options{2,2} = single(10^(-6));
  options{3,1}='PSVDtol';
  options{3,2}=single(.01);
  options{4,1}='PSVDmaxiter';
  options{4,2}=int32(500);
  
  for aa=3:3
    algstr=alg_list{aa};

  tic
  [gpu_norms gpu_times gpu_iter gpu_conv gpu_MatOut gpu_MatInput gpu_A] = gagamc_entry_suic(algstr,int32(m), int32(n), int32(r), int32(p),options); 
  algstr, gpu_norms, gpu_times, gpu_iter
  toc
  

  end

end

%total_time = toc(stime)
