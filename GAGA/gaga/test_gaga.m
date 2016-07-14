% file to run each of the gaga functions once and print support_check

alg_list=cell(7,1);
alg_list{1}= 'NIHT';
alg_list{2}= 'HTP';
alg_list{3}= 'CSMPSP';
alg_list{4}= 'FIHT';
alg_list{5}= 'CGIHT';
alg_list{6}= 'CGIHTprojected';
alg_list{7}= 'CGIHT';

ens_list=cell(3,1);
ens_list{1}='gen';
ens_list{2}='dct';
ens_list{3}='smv';

delta = 0.5;
rho = 0.25;

norms=zeros(length(alg_list),3);
timer=norms;

tic;

for i=1:length(ens_list)
  if strcmp(ens_list{i},'gen')
    n=2^(14);
  elseif strcmp(ens_list{i},'dct')
    n=2^(14);
  elseif strcmp(ens_list{i},'smv')
    p=7;
    n=2^(13);
  end
  m=ceil(n*delta);
  k=ceil(m*rho);
  % testing without noise 
  options=gagaOptions('maxiter',805,'tol',10^(-7),'noise',0.0,'gpuNumber',0,'kFixed','on','restartFlag','off','projFracTol',4);
  for j=1:length(alg_list)
   if j==7
     options=gagaOptions('maxiter',805,'tol',10^(-7),'noise',0.0,'gpuNumber',0,'kFixed','on','restartFlag','on','projFracTol',4);
   end
display(sprintf('This is %s. \n \n',alg_list{j}))
    if strcmp(ens_list{i},'gen')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'dct')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'smv')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, p, options);
    end
  end
  norms
  timer

  % testing with noise 

  options=gagaOptions('maxiter',805,'tol',10^(-7),'noise',0.10,'gpuNumber',0,'kFixed','on','restartFlag','off','projFracTol',4);
  for j=1:length(alg_list)
   if j==7
     options=gagaOptions('maxiter',805,'tol',10^(-7),'noise',0.10,'gpuNumber',0,'kFixed','on','restartFlag','on','projFracTol',4);
   end
    if strcmp(ens_list{i},'gen')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'dct')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'smv')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, p, options);
    end
  end
  norms
  timer
%{
  % testing with extra timing output
  options=gagaOptions('maxiter',50,'tol',10^(-4),'noise',0.0,'gpuNumber',0,'timing','on');
  for j=1:length(alg_list)
    if strcmp(ens_list{i},'gen')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'dct')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, options);
    elseif strcmp(ens_list{i},'smv')
      [norms(j,:) timer(j,:) c d e f] = gaga_cs(alg_list{j}, ens_list{i}, k, m, n, p, options);
    end
  end
  
  support = randperm(n);
  x = sign(randn(n,1));
  x(support(k+1:end))= 0.*x(support(k+1:end));
  
  if strcmp(ens_list{i},'dct')
    rr=randperm(n);
    dct_rows = rr(1:m);

    dct_y = A_dct(x, m, n, dct_rows);

    dct_y = single(dct_y);
    dct_rows = int32(dct_rows);
  elseif strcmp(ens_list{i},'smv')    
    p=9;
    smv_rows = ceil(m*rand(n*p,1));
    smv_cols = zeros(n*p,1);
    for jj=1:n
      smv_cols((jj-1)*p+1:jj*p)=jj*ones(p,1);
    end
    smv_vals = sign(randn(n*p,1))/sqrt(p);


    A_smv = sparse(smv_rows, smv_cols, smv_vals);
    smv_y = A_smv*x;

    smv_y = single(smv_y);
    smv_rows = int32(smv_rows);
    smv_cols = int32(smv_cols);
    smv_vals = single(smv_vals);
  elseif strcmp(ens_list{i},'gen')
    A_gen = randn(m,n)/sqrt(m);
    gen_y = A_gen*x;

    A_gen = single(A_gen);
    gen_y = single(gen_y);
  end
  
  options=gagaOptions('maxiter',800,'tol',10^(-4),'noise',0.0,'gpuNumber',0,'kFixed','on');
  for j=1:length(alg_list)
    if strcmp(ens_list{i},'gen')
      [norms(j,:) timer(j,:) c] = gaga_cs(alg_list{j}, ens_list{i}, gen_y, A_gen, k, options);
      display(sprintf('Error for %s with %s: %f', alg_list{j}, ens_list{i}, norm(a-x,inf)));
    elseif strcmp(ens_list{i},'dct')
      [norms(j,:) timer(j,:) c] = gaga_cs(alg_list{j}, ens_list{i}, dct_y, dct_rows, k, m, n, options);
      display(sprintf('Error for %s with %s: %f', alg_list{j}, ens_list{i}, norm(a-x,inf)));
    elseif strcmp(ens_list{i},'smv')
      [norms(j,:) timer(j,:) c] = gaga_cs(alg_list{j}, ens_list{i}, smv_y, smv_rows, smv_cols, smv_vals, k, options);
      display(sprintf('Error for %s with %s: %f', alg_list{j}, ens_list{i}, norm(a-x,inf)));
    end
  end
  %}
end

display(sprintf('The tests completed after %s seconds.',toc));
