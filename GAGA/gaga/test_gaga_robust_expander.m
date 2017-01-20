% file to run each of the gaga functions once and print support_check

alg_list=cell(4,1);
alg_list{1}= 'smp';
alg_list{2}= 'ssmp';
alg_list{3}= 'robust_l0';
alg_list{4}= 'deterministic_robust_l0';
alg_list{5}= 'adaptive_robust_l0';

ens_list=cell(1,1);
ens_list{1}='smv';

delta = 0.5;
rho = 0.03;

noise_level = 1e-2;
tolerance = 1e-3;
l0_thresh = 2;

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
    n=2^(14);
  end
  m=ceil(n*delta);
  k=ceil(m*rho);

  % testing without noise 
  options=gagaOptions('maxiter',3*k,'tol',tolerance,'noise',noise_level,'gpuNumber',0,'vecDistribution','gaussian','kFixed','on', 'l0_thresh', int32(l0_thresh));
  for j=1:length(alg_list)
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

  % Testing by feeding a problem to the algorithms

  support = randperm(n);
  x = randn(n,1);
  x(support(k+1:end))= 0.*x(support(k+1:end));
  
  if 0 && strcmp(ens_list{i},'smv')    
    p=9;
    smv_rows = ceil(m*rand(n*p,1));
    smv_cols = zeros(n*p,1);
    for jj=1:n
      smv_cols((jj-1)*p+1:jj*p)=jj*ones(p,1);
    end
    smv_vals = ones(n*p,1);

    A_smv = sparse(smv_rows, smv_cols, smv_vals);
    smv_y = A_smv*x;

    smv_y = single(smv_y);
    smv_rows = int32(smv_rows);
    smv_cols = int32(smv_cols);
    smv_vals = single(smv_vals);
  
    options=gagaOptions('maxiter',3*k,'tol',tolerance,'noise',noise_level,'gpuNumber',0,'kFixed','on', 'l0_thresh', int32(l0_thresh));
    for j=1:length(alg_list)
        [a b c] = gaga_cs(alg_list{j}, ens_list{i}, smv_y, smv_rows, smv_cols, smv_vals, k, options);
        display(sprintf('Error for %s with %s: %f', alg_list{j}, ens_list{i}, norm(a-x,inf)));
    end
  end

end

display(sprintf('The tests completed after %s seconds.',toc));
