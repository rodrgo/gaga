% file to run each of the gaga functions once and print support_check

alg_list=cell(5,1);
alg_list{1}= 'SMP';
alg_list{2}= 'SSMP';
alg_list{3}= 'ER';
alg_list{4}= 'serial-l0';
alg_list{5}= 'parallel-l0';

ens_list=cell(1,1);
ens_list{1}='smv';

delta = 0.5;
rho = 0.1;

for i=1:length(ens_list)
  if strcmp(ens_list{i},'gen')
    n=2^(12);
  elseif strcmp(ens_list{i},'dct')
    n=2^(18);
  elseif strcmp(ens_list{i},'smv')
    p=7;
    n=2^(16);
  end
  m=ceil(n*delta);
  k=ceil(m*rho);
  % testing without noise 
  options=gagaOptions('maxiter',2*k,'tol',10^(-7), 'matrixEnsemble', 'ones', 'vecDistribution', 'gaussian', 'noise',0.0);
  for j=1:length(alg_list)
    if strcmp(ens_list{i},'gen')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'dct')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'smv')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, p, options);
    end
  end
  
end


