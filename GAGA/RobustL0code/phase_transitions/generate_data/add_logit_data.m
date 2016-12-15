% Function modified on 12/05/14 by RMS
% alg name now enters in full (with ensemble
% data appended to it)
% ensemble data is entered redundantly to
% facilitate pipelining.

function add_logit_data(alg, ensemble, SOL_TOL)

% One results file per algorithm
% (may have data for various noise levels)

fname=['results_' alg];
load(fname)

for cell_num = 1:length(results_cell)

  % Extract data from cell

  results = results_cell{cell_num}.results;
  columns = results_cell{cell_num}.columns;  
  noise_level = results_cell{cell_num}.noise_level;

  % Keep going as before 

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
    elseif strcmp(ensemble,'smv')
      ind_linfty=9;
    elseif strcmp(ensemble,'gen')
      ind_linfty=8;
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

          if any(ismember({'deterministic_robust_l0', 'robust_l0', 'ssmp_robust'}, {alg})) 
            mean_err1 = m_list(i)*noise_level*sqrt(2/pi); 
            sd_err1 = sqrt(m_list(i))*noise_level*sqrt(1 - 2/pi); 
		mean_signal_norm = k_list(j)*sqrt(2/pi);
            success=[success; sum(results_tmp(ind,7) <= (mean_err1 + SOL_TOL*sd_err1)/mean_signal_norm)/length(ind)];
          else
            % results_tmp(ind, 8) is the l2 norm
            success=[success; sum(results_tmp(ind,8)<=SOL_TOL)/length(ind)];
          end
          %success=[success; sum(results_tmp(ind,ind_linfty)<0.01)/length(ind)];
          k=[k; k_list(j)];
          m=[m; m_list(i)];
        end
      end
    end

    beta_list=zeros(length(m_list),2);
    error_list=zeros(length(m_list),1);

    for i=1:length(m_list)
      ind=find(m==m_list(i));
      b=calc_logit(k(ind),m(ind),success(ind),num_tests(ind));
      beta_list(i,:)=b;
      error_list(i)=logit_model_error(k(ind)./m(ind),success(ind),num_tests(ind),b);
    end
    
    betas{nn}=beta_list;
    deltas{nn}=m_list/n;

    % Store variables in results_cell 

    if strcmp(ensemble,'smv')
      results_cell{cell_num}.betas = betas;
      results_cell{cell_num}.deltas = deltas;
      results_cell{cell_num}.n_list = n_list;
      results_cell{cell_num}.nz_list = nz_list;
    else
      results_cell{cell_num}.betas = betas;
      results_cell{cell_num}.deltas = deltas;
      results_cell{cell_num}.n_list = n_list;
    end
    
  end

end

save(fname,'results_cell');

