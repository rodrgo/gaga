% file to run each of the gaga functions once and print support_check

alg_list=cell(4,1);
alg_list{1}= 'NIHT';
alg_list{2}= 'IHT';
alg_list{3}= 'HTP';
alg_list{4}= 'CSMPSP';

alg_timing_list=cell(2,1);
alg_timing_list{1}= 'NIHT';
alg_timing_list{2}= 'HTP';


ens_list=cell(3,1);
ens_list{1}='gen';
ens_list{2}='dct';
ens_list{3}='smv';

delta = 0.5;
rho = 0.2;

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
  options=gagaOptions('maxiter',800,'tol',10^(-7),'noise',0.0);
  for j=1:length(alg_list)
    if strcmp(ens_list{i},'gen')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'dct')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'smv')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, p, options);
    end
  end
  % testing with noise 
  options=gagaOptions('maxiter',200,'tol',10^(-4),'noise',0.1,'timing','off');
  for j=1:length(alg_list)
    if strcmp(ens_list{i},'gen')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'dct')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'smv')
      [a b c d e f] = gaga_matlab_cs(alg_list{j}, ens_list{i}, k, m, n, p, options);
    end
  end
  % testing with extra timing output
  options=gagaOptions('maxiter',50,'tol',10^(-4),'noise',0.0,'timing','on');
  for j=1:length(alg_timing_list)
    if strcmp(ens_list{i},'gen')
      [a b c d e f] = gaga_matlab_cs(alg_timing_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'dct')
      [a b c d e f] = gaga_matlab_cs(alg_timing_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'smv')
      [a b c d e f] = gaga_matlab_cs(alg_timing_list{j}, ens_list{i}, k, m, n, p, options);
    end
  end
  
end


