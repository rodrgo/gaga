gpuNum=0;

noise_level_list = [0 0.1  0.2];

tttt=tic;

for nl=1:length(noise_level_list)
noise_level=noise_level_list(nl);

tolerance = .0001;  %+.1*noise_level;

maxIter=500;

options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','on');

matens_list = cell(1,1);
matens_list{1} ='dct';
%matens_list{1} ='gen';
%matens_list{2} ='smv';

ttttttt=tic;

for w=1:length(matens_list)
matens = matens_list{w};


NIHT_CGIHT_CGrestart_results=[];
NIHT_CGIHT_CGrestart_iter=[];
NIHT_CGIHT_CGrestart_maxiter=[];
NIHT_CGIHT_CGrestart_supp=[];
for qq=13:2:13; %2:12
m=2^(qq-2);
n=2^qq;
m=round(.5*n)

nonzeros=7;

rho=linspace(0.05,0.2,4); rho=rho';
k_list=round(rho*m);

rho=0.275;
k_list=round(rho*m);

num_tests=1;


sofar=[w n]

time_niht=zeros(length(k_list),num_tests);
rate_niht=zeros(length(k_list),num_tests);
num_iter_niht=zeros(length(k_list),num_tests);
error_linf_niht=zeros(length(k_list),num_tests);
suppfrac_niht=zeros(length(k_list),num_tests);

time_cgiht=zeros(length(k_list),num_tests);
rate_cgiht=zeros(length(k_list),num_tests);
num_iter_cgiht=zeros(length(k_list),num_tests);
error_linf_cgiht=zeros(length(k_list),num_tests);
suppfrac_cgiht=zeros(length(k_list),num_tests);

time_restarted=zeros(length(k_list),num_tests);
rate_restarted=zeros(length(k_list),num_tests);
num_iter_restarted=zeros(length(k_list),num_tests);
error_linf_restarted=zeros(length(k_list),num_tests);
suppfrac_restarted=zeros(length(k_list),num_tests);

%tic
for p=1:length(k_list)
  k = k_list(p);
%sofar=[n p]
%{
  for q=1:num_tests
   if strcmp(matens,'smv')

    [err_niht t_niht iter_niht d r_niht f] = gaga_cs('NIHT', matens, k,m,n, nonzeros, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','on');
    [err_restarted t_restarted iter_restarted d r_restarted f] = gaga_cs('CGIHT', matens, k,m,n, nonzeros, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','off');
    [err_cgiht t_cgiht iter_cgiht d r_cgiht f] = gaga_cs('CGIHT', matens, k,m,n, nonzeros, options); 

   else 

    [err_niht t_niht iter_niht d r_niht f] = gaga_cs('NIHT', matens, k,m,n, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','on');
    [err_restarted t_restarted iter_restarted d r_restarted f] = gaga_cs('CGIHT', matens, k,m,n, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','off');
    [err_cgiht t_cgiht iter_cgiht d r_cgiht f] = gaga_cs('CGIHT', matens, k,m,n, options); 

   end
%}
 for q=1:num_tests
  if strcmp(matens,'gen')
    [err_niht t_niht iter_niht supp_niht r_niht f] = gaga_cs('NIHT', 'gen', k,m,n, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','on');
    [err_restarted t_restarted iter_restarted supp_restarted r_restarted f] = gaga_cs('CGIHT', 'gen', k,m,n, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','off');
    [err_cgiht t_cgiht iter_cgiht supp_cgiht r_cgiht f] = gaga_cs('CGIHT', 'gen', k,m,n, options);    
  elseif strcmp(matens, 'dct')

    [err_niht t_niht iter_niht supp_niht r_niht f] = gaga_cs('NIHT', 'dct', k,m,n, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','on');
    [err_restarted t_restarted iter_restarted supp_restarted r_restarted f] = gaga_cs('CGIHT', 'dct', k,m,n, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','off');
    [err_cgiht t_cgiht iter_cgiht supp_cgiht r_cgiht f] = gaga_cs('CGIHT', 'dct', k,m,n, options);  
  elseif strcmp(matens, 'smv')

    [err_niht t_niht iter_niht supp_niht r_niht f] = gaga_cs('NIHT', 'smv', k,m,n,7, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','on');
    [err_restarted t_restarted iter_restarted supp_restarted r_restarted f] = gaga_cs('CGIHT', 'smv', k,m,n,7, options);

    options=gagaOptions('maxiter',maxIter,'tol',tolerance,'noise',noise_level,'gpuNumber',gpuNum,'kFixed','on','restartFlag','off');
    [err_cgiht t_cgiht iter_cgiht supp_cgiht r_cgiht f] = gaga_cs('CGIHT', 'smv', k,m,n,7, options); 
  end  % end if matens

%[(p-1)*10+q single(supp_niht(1))/single(sum(supp_niht(1:2))) single(supp_restarted(1))/single(sum(supp_restarted(1:2))) single(supp_cgiht(1))/single(sum(supp_cgiht(1:2)))]
%[p q err_niht err_restarted err_cgiht]
    time_niht(p,q)=t_niht(1);
    rate_niht(p,q)=r_niht;
    num_iter_niht(p,q)=iter_niht;
    error_linf_niht(p,q)=err_niht(2);
    suppfrac_niht(p,q)=single(supp_niht(1))/single(sum(supp_niht(1:2)));

    time_cgiht(p,q)=t_cgiht(1);
    rate_cgiht(p,q)=r_cgiht;
    num_iter_cgiht(p,q)=iter_cgiht;
    error_linf_cgiht(p,q)=err_cgiht(2);
    suppfrac_cgiht(p,q)=single(supp_cgiht(1))/single(sum(supp_cgiht(1:2)));

    time_restarted(p,q)=t_restarted(1);
    rate_restarted(p,q)=r_restarted;
    num_iter_restarted(p,q)=iter_restarted;
    error_linf_restarted(p,q)=err_restarted(2);
    suppfrac_restarted(p,q)=single(supp_restarted(1))/single(sum(supp_restarted(1:2)));

%    [rate_niht(p,q) rate_cgiht(p,q)]


  end  % end for q=1:num_tests
%  [p k/m toc]
%  save test_cgiht_data.mat
end % end for p=1:length(k_list)

  
format short g
tol=10^(-3)+2*noise_level;
%display('n k p_niht p_cgiht p_restarted t_niht t_cgiht t_restarted')
%if qq>9
NIHT_CGIHT_CGrestart_results = [NIHT_CGIHT_CGrestart_results; n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2) sum(time_niht,2)/num_tests sum(time_cgiht,2)/num_tests sum(time_restarted,2)/num_tests ];
NIHT_CGIHT_CGrestart_iter = [NIHT_CGIHT_CGrestart_iter; n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2)  sum(num_iter_niht,2)/num_tests sum(num_iter_cgiht,2)/num_tests sum(num_iter_restarted,2)/num_tests];
%NIHT_CGIHT_CGrestart_maxiter = [NIHT_CGIHT_CGrestart_maxiter; n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2)  max(num_iter_niht')' max(num_iter_cgiht')' max(num_iter_restarted')'];
%NIHT_CGIHT_CGrestart_supp = [NIHT_CGIHT_CGrestart_supp; n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2)  min(suppfrac_niht')' min(suppfrac_cgiht')' min(suppfrac_restarted')'];

NIHT_CGIHT_CGrestart_supp = [NIHT_CGIHT_CGrestart_supp; n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2)  sum(suppfrac_niht,2)/num_tests sum(suppfrac_cgiht,2)/num_tests sum(suppfrac_restarted,2)/num_tests];


%NIHT_CGIHT_CGrestart_results = [NIHT_CGIHT_CGrestart_results; n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2) sum(time_niht,2)/num_tests sum(time_cgiht,2)/num_tests sum(time_restarted,2)/num_tests ];

%else
%NIHT_CGIHT_CGrestart_results = [n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2) sum(time_niht,2)/num_tests sum(time_cgiht,2)/num_tests sum(time_restarted,2)/num_tests ];
%end
end % end for qq (problem size)

%acc=[NIHT_CGIHT_CGrestart_results(:,6)./NIHT_CGIHT_CGrestart_results(:,7) NIHT_CGIHT_CGrestart_results(:,6)./NIHT_CGIHT_CGrestart_results(:,8) NIHT_CGIHT_CGrestart_results(:,7)./NIHT_CGIHT_CGrestart_results(:,8)];
display(sprintf('Matens = %s, noise = %0.2f',matens,noise_level))
timings=[NIHT_CGIHT_CGrestart_results]
%iterations=[NIHT_CGIHT_CGrestart_iter]
%maxiterations=[NIHT_CGIHT_CGrestart_maxiter]
avesuppfrac=[NIHT_CGIHT_CGrestart_supp]
avesiter=[NIHT_CGIHT_CGrestart_iter]

%resultswithratios=timings;
%resultswithratios(:,end-2:end)=acc;

%display('n k p_niht p_cgiht p_restarted t_niht/t_cgiht t_niht/t_restarted t_cgiht/t_restarted')
%timingratios=resultswithratios

end % end for w=1:length(matens_list)

totaltestingtime = toc(ttttttt)

end % end noise_level_list


totaltestingtime = toc(tttt)

