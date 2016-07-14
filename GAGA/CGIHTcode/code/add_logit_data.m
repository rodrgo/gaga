function add_logit_data(alg,ensemble)
% add_logit_data(alg,ensemble) computes the logistic regression
% for the given algorithm and matrix ensemble for the problem class
% (Mat, B) with no noise.  It uses the l_infty error.
% For data with noise or for other vector distrubtions with or without noise,
% use the function add_logit_data_noise.


addpath ../shared/

fname=['results_' alg '_S_' ensemble '.mat'];
load(fname)


format shortg
min_tests=1;

n_list_tmp=intersect(results(:,3),results(:,3));

n_list=[];
nz_list=[];

if strcmp(ensemble,'smv')
  for nn=1:length(n_list_tmp)

    n=n_list_tmp(nn);
    ind=find(results(:,3)==n);
    nz=intersect(results(ind,5),results(ind,5));
    
    n_list=[n_list; n*ones(length(nz),1)];
    nz_list=[nz_list; nz];
  end
else
  n_list=n_list_tmp;
end


betas=cell(length(n_list),1);
deltas=cell(length(n_list),1);

for nn=1:length(n_list)

  n=n_list(nn);
  if strcmp(ensemble,'smv')
    ind=find(results(:,3)==n & results(:,5)==nz_list(nn));
  else
    ind=find(results(:,3)==n);
  end
  
  results_tmp=results(ind,:);

  m_list=intersect(results_tmp(:,2),results_tmp(:,2));


  
  k=[];
  m=[];
  success=[];
  num_tests=[];
  
  
  if strcmp(ensemble,'dct')
    ind_linfty=7;
    ind_ltwo = 6;
  elseif strcmp(ensemble,'smv')
    ind_linfty=9;
    ind_ltwo = 8;
  elseif strcmp(ensemble,'gen')
    ind_linfty=8;
    ind_ltwo = 7;
  end

  for i=1:length(m_list)
    m_tmp=m_list(i);
    ind=find(results_tmp(:,2)==m_tmp);
    k_tmp=results_tmp(ind,1);
    k_list=intersect(k_tmp,k_tmp);
  
    for j=1:length(k_list)
      ind=find(results_tmp(:,1)==k_list(j) & results_tmp(:,2)==m_list(i));
      if length(ind)>=min_tests
        num_tests=[num_tests; length(ind)];
        success=[success; sum(results_tmp(ind,ind_linfty)< (0.01))/length(ind)];
        k=[k; k_list(j)];
        m=[m; m_list(i)];
      end
    end
  end

  beta_list=zeros(length(m_list),2);
  error_list=zeros(length(m_list),1);

  tic
  for i=1:length(m_list)
    ind=find(m==m_list(i));
    b=calc_logit(k(ind),m(ind),success(ind),num_tests(ind));
    beta_list(i,:)=b;
    error_list(i)=logit_model_error(k(ind)./m(ind),success(ind),num_tests(ind),b);
%    hold off
%    plot(k(ind)./m(ind),success(ind),'ro')
%    hold on
%    plot(k(ind)./m(ind),logit_model(k(ind)./m(ind),b))
%    hold off
%    beta_list(i,:)
%    [i length(m_list) error_list(i)]
%    toc
%    pause
  end
  
  betas{nn}=beta_list;
  deltas{nn}=m_list/n;
  
end
%   [beta_list error_list 1./beta_list(:,2)]
%   plot(m_list/n,1./beta_list(:,2))


if strcmp(ensemble,'smv')
  save(fname,'results','columns','betas','deltas','n_list','nz_list')
else
  save(fname,'results','columns','betas','deltas','n_list')
end

