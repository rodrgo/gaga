options=gagaOptions('maxiter',800,'tol',10^(-5),'noise',0.0,'restartFlag','on');

matens_list = cell(3,1);
matens_list{1} ='dct';
matens_list{2} ='gen';
matens_list{3} ='smv';

ttttttt=tic;

for w=3:3
matens = matens_list(w);

NIHT_CGIHT_CGrestart_results=[];
for qq=12 %8:2:12
m=2^(qq-1);
n=2^qq;

nonzeros=7;

rho=linspace(0.01,0.4,15); rho=rho';
k_list=round(rho*m);

%rho=0.01;
%k_list=round(rho*m);

num_tests=100;


sofar=[w n]

time_niht=zeros(length(k_list),num_tests);
rate_niht=zeros(length(k_list),num_tests);
num_iter_niht=zeros(length(k_list),num_tests);
error_linf_niht=zeros(length(k_list),num_tests);

time_cgiht=zeros(length(k_list),num_tests);
rate_cgiht=zeros(length(k_list),num_tests);
num_iter_cgiht=zeros(length(k_list),num_tests);
error_linf_cgiht=zeros(length(k_list),num_tests);

time_restarted=zeros(length(k_list),num_tests);
rate_restarted=zeros(length(k_list),num_tests);
num_iter_restarted=zeros(length(k_list),num_tests);
error_linf_restarted=zeros(length(k_list),num_tests);

tic
for p=1:length(k_list)
  k = k_list(p);
%sofar=[n p]
  for q=1:num_tests
   if strcmp(matens,'smv')

    [err_niht t_niht iter_niht d r_niht f] = gaga_matlab_cs('NIHT', matens, k,m,n, nonzeros, options);

    options=gagaOptions('maxiter',800,'tol',10^(-4),'noise',0.0,'restartFlag','on');
    [err_restarted t_restarted iter_restarted d r_restarted f] = gaga_matlab_cs('CGIHT', matens, k,m,n, nonzeros, options);

    options=gagaOptions('maxiter',800,'tol',10^(-4),'noise',0.0,'restartFlag','off');
    [err_cgiht t_cgiht iter_cgiht d r_cgiht f] = gaga_matlab_cs('CGIHT', matens, k,m,n, nonzeros, options); 

   else 

    [err_niht t_niht iter_niht d r_niht f] = gaga_matlab_cs('NIHT', matens, k,m,n, options);

    options=gagaOptions('maxiter',800,'tol',10^(-5),'noise',0.0,'restartFlag','on');
    [err_restarted t_restarted iter_restarted d r_restarted f] = gaga_matlab_cs('CGIHT', matens, k,m,n, options);

    options=gagaOptions('maxiter',800,'tol',10^(-5),'noise',0.0,'restartFlag','off');
    [err_cgiht t_cgiht iter_cgiht d r_cgiht f] = gaga_matlab_cs('CGIHT', matens, k,m,n, options); 

   end


    time_niht(p,q)=t_niht(1);
    rate_niht(p,q)=r_niht;
    num_iter_niht(p,q)=iter_niht;
    error_linf_niht(p,q)=err_niht(3);

    time_cgiht(p,q)=t_cgiht(1);
    rate_cgiht(p,q)=r_cgiht;
    num_iter_cgiht(p,q)=iter_cgiht;
    error_linf_cgiht(p,q)=err_cgiht(3);

    time_restarted(p,q)=t_restarted(1);
    rate_restarted(p,q)=r_restarted;
    num_iter_restarted(p,q)=iter_restarted;
    error_linf_restarted(p,q)=err_restarted(3);

%    [rate_niht(p,q) rate_cgiht(p,q)]


  end
%  [p k/m toc]
%  save test_cgiht_data.mat
end

  
format short g
tol=10^(-3);
display('n k p_niht p_cgiht p_restarted t_niht t_cgiht t_restarted')
%if qq>9
NIHT_CGIHT_CGrestart_results = [NIHT_CGIHT_CGrestart_results; n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2) sum(time_niht,2)/num_tests sum(time_cgiht,2)/num_tests sum(time_restarted,2)/num_tests ];

%else
%NIHT_CGIHT_CGrestart_results = [n*ones(size(k_list)) k_list sum(error_linf_niht<tol,2) sum(error_linf_cgiht<tol,2) sum(error_linf_restarted<tol,2) sum(time_niht,2)/num_tests sum(time_cgiht,2)/num_tests sum(time_restarted,2)/num_tests ];
%end
end

%acc=[NIHT_CGIHT_CGrestart_results(:,6)./NIHT_CGIHT_CGrestart_results(:,7) NIHT_CGIHT_CGrestart_results(:,6)./NIHT_CGIHT_CGrestart_results(:,8) NIHT_CGIHT_CGrestart_results(:,7)./NIHT_CGIHT_CGrestart_results(:,8)];

timings=[NIHT_CGIHT_CGrestart_results]

%resultswithratios=timings;
%resultswithratios(:,end-2:end)=acc;

%display('n k p_niht p_cgiht p_restarted t_niht/t_cgiht t_niht/t_restarted t_cgiht/t_restarted')
%timingratios=resultswithratios

end 

totaltestingtime = toc(ttttttt)

