num_tests = 10;
tol = 10^(-6);


sd_recovery = 0;
cg_recovery = 0;
cgr_recovery = 0;

cgr_restart_flag = 1; % when equal to one we resort to NIHT,
                      % otherwise use CGIHT on a restricted column space.


show_plots = 1;

for jj = 1 : num_tests
  m = 100;
  n = 400;
  
  %delta = 0.3;
  %rho = 0.7;
  
  %p = round(delta*m*n);
  %rank = round(rho*p/(m+n));
  p = 32000;
  r = 30;

% set the parameter we call zeta in the paper  
  %cgr_restart_tol = 100 %(r/max(m,n))*1
  
  num_steps = 300;
  
  iter_of_restarts=zeros(num_steps,1);
  
  norm_e_sd = zeros(num_steps,1);
  norm_r_sd = zeros(num_steps,1);
  t_sd = zeros(num_steps,1);
  
  norm_e_cg = zeros(num_steps,1);
  norm_r_cg = zeros(num_steps,1);
  t_cg = zeros(num_steps,1);
 
  norm_e_cgr = zeros(num_steps,1);
  norm_r_cgr = zeros(num_steps,1);
  t_cgr = zeros(num_steps,1);
  
  A = zeros(m*n,1);
  A(1:p) = ones(p,1);
  [~,ind] = sort(randn(m*n,1));
  A = A(ind);
  A = reshape(A,m,n);

  L = randn(m,r);
  R = randn(r,n);
  M = L * R;
  data = A.*M;
  
  res_denom = norm(data, 'fro');
  err_denom = norm(M,'fro');
  
  % initial point 
  [U, S, V] = svds(A.*M,r);
  
  sd_M = U*S*V';
  cg_M = U*S*V';
  cgr_M = U*S*V';
  
  cgr_U=U;
  
  sd_grad = data - A.*sd_M;
  cg_grad = data - A.*cg_M;
  cgr_grad = data - A.*cgr_M;
  
  sd_grad_proj = U*(U'*sd_grad);
  
  cg_search_dir = cg_grad;
  cg_grad_proj = U*(U'*cg_grad);
  cg_search_dir_proj = U*(U'*cg_search_dir);
  
  cgr_search_dir = cgr_grad;
  cgr_grad_proj = U*(U'*cgr_grad);
  cgr_search_dir_proj = U*(U'*cgr_search_dir);
  
  sd_res = norm(sd_grad,'fro')/res_denom;
  sd_err = norm(sd_M-M,'fro')/err_denom;
  
  cg_res = norm(cg_grad,'fro')/res_denom;
  cg_err = norm(cg_M-M,'fro')/err_denom;
  
  cgr_res = norm(cgr_grad,'fro')/res_denom;
  cgr_err = norm(cgr_M-M,'fro')/err_denom;
  
  sd_active = 1;
  cg_active = 1;
  cgr_active = 1;
  
  sd_iter=num_steps;
  cg_iter=num_steps;
  cgr_iter=num_steps;
  
  for j = 1 : num_steps
    % niht for matrix completion
    tic;
    if (sd_active == 1)
      norm_e_sd(j) = sd_err;
      norm_r_sd(j) = sd_res;
      
      sd_alpha = norm(sd_grad_proj,'fro')^2;
      tmp = norm(A.*sd_grad_proj,'fro')^2;
      if abs(sd_alpha) < 1000*tmp
        sd_alpha = sd_alpha/tmp;
      else
        sd_alpha = 1;
      end
           
      sd_M = sd_M + sd_alpha * sd_grad;
      
      [sd_U, sd_S, sd_V] = svds(sd_M,r);
      sd_M = sd_U*sd_S*sd_V';
      
      sd_grad = data - A.*sd_M;
      
      sd_res = norm(sd_grad,'fro')/res_denom;
      sd_err = norm(sd_M-M, 'fro')/err_denom;
      
      sd_grad_proj = sd_U*(sd_U'*sd_grad);
      
      t_sd(j) = toc; 
     
      if (sd_res< tol)
        sd_active = 0;
        sd_iter = j;
      end
    end % end of niht

    % cgiht for matrix completion (uncgr)
    if (cg_active ==1)
      tic;
      norm_e_cg(j) = cg_err;
      norm_r_cg(j) = cg_res;
      
      cg_alpha = sum(sum(cg_search_dir_proj.*cg_grad_proj));           
      tmp = norm(A.*cg_search_dir_proj,'fro')^2;
      if abs(cg_alpha) < 1000*tmp;
        cg_alpha = cg_alpha/tmp;
      else
        cg_alpha = 1;
        display('CGIHT had alpha set to 1 artificially')
      end
      
      cg_M = cg_M + cg_alpha * cg_search_dir;
      
      [cg_U, cg_S, cg_V] = svds(cg_M, r);
      cg_M = cg_U*cg_S*cg_V';
      
      cg_grad = data - A.*cg_M;
      
      cg_res  = norm(cg_grad,'fro')/res_denom;
      cg_err = norm(cg_M - M, 'fro')/err_denom;
      
      cg_grad_proj = cg_U*(cg_U'*cg_grad);
      cg_search_dir_proj = cg_U*(cg_U'*cg_search_dir);
      
      Acg_grad_proj = A.*cg_grad_proj;
      Acg_search_dir_proj = A.*cg_search_dir_proj;
      
      cg_beta = sum(sum(A.*cg_grad_proj.*cg_search_dir_proj))...
                / sum(sum(A.*cg_search_dir_proj.*cg_search_dir_proj));
      
      
      cg_search_dir = cg_grad - cg_beta * cg_search_dir;
      
      % compute cg_search_dir_proj
      cg_search_dir_proj = cg_U*(cg_U'*cg_search_dir);
      
      t_cg(j) = toc;
      
      if (cg_res < tol)
        cg_active = 0;
        cg_iter = j;
      end
    end % end of cgiht
    
    
    
    % cgiht for matrix completion (cgr)
    if (cgr_active ==1)
      tic;
      norm_e_cgr(j) = cgr_err;
      norm_r_cgr(j) = cgr_res;
      
      cgr_alpha = sum(sum(cgr_search_dir_proj.*cgr_grad_proj));
      tmp = norm(A.*cgr_search_dir_proj,'fro')^2;
      if abs(cgr_alpha) < 1000*tmp;
        cgr_alpha = cgr_alpha/tmp;
      else
        cgr_alpha = 1;
        display('cgr had alpha set to 1 artificially')
      end
      
      cgr_M = cgr_M + cgr_alpha * cgr_search_dir;
      
      [cgr_U, cgr_S, cgr_V] = svds(cgr_M,r);
      cgr_M = cgr_U*cgr_S*cgr_V';
      
      % weighted orthogonal to previous residual
      cgr_search_dir = cgr_grad; 
      
      cgr_grad = data - A.*cgr_M;
      
      cgr_res  = norm(cgr_grad,'fro')/res_denom;
      cgr_err = norm(cgr_M - M, 'fro')/err_denom;
      
      cgr_grad_proj = cgr_U*(cgr_U'*cgr_grad);
      
      % one step niht followed by one step cgiht
      if mod(j,2) == 1
        cgr_search_dir = cgr_grad;
        cgr_search_dir_proj = cgr_grad_proj;
      else
        cgr_search_dir_proj = cgr_U*(cgr_U'*cgr_search_dir);
        cgr_beta = sum(sum(A.*cgr_grad_proj.*cgr_search_dir_proj));
        tmp = sum(sum(A.*cgr_search_dir_proj.*cgr_search_dir_proj));
        if abs(cgr_beta) < 1000*tmp
          cgr_beta = cgr_beta/tmp;
        else
          cgr_beta = 0;
        end     
        cgr_search_dir = cgr_grad - cgr_beta * cgr_search_dir;
        cgr_search_dir_proj = cgr_grad_proj - cgr_beta * cgr_search_dir_proj;
      end
      
      
      t_cgr(j) = toc;
      
      if (cgr_res < tol)
        cgr_active = 0;
        cgr_iter = j;
        %cgr_iter
      end
    end % end of cgriht
    
    

    if (sd_active == 0 && cg_active == 0 && cgr_active == 0)
      break;
    end
  end % end of all algorithms for one test
  
  % note all res and err are relative 

  if (show_plots == 1)
    figure(1)
    hold off
    semilogy(norm_e_sd(1:sd_iter),'r')
    hold on
    semilogy(norm_r_sd(1:sd_iter), 'r--')    
    semilogy(norm_e_cg(1:cg_iter), 'k')
    semilogy(norm_r_cg(1:cg_iter), 'k--')
    semilogy(norm_e_cgr(1:cgr_iter), 'b')
    semilogy(norm_r_cgr(1:cgr_iter), 'b--')
    
    legend('e\_sd', 'r\_sd', 'e\_cg', 'r\_cg', 'e\_cgr', 'r\_cgr')
  end
  
  for kk = 2 : sd_iter
    t_sd(kk) = t_sd(kk) + t_sd(kk-1);
  end
  
  for kk = 2 : cg_iter
    t_cg(kk) = t_cg(kk) + t_cg(kk-1);
  end
  
  for kk = 2 : cgr_iter
    t_cgr(kk) = t_cgr(kk) + t_cgr(kk-1);
  end
  
  if (show_plots == 1)
    figure(2)
    hold off
    semilogy(t_sd(1:sd_iter), norm_e_sd(1:sd_iter), 'r')
    hold on
    semilogy(t_cg(1:cg_iter), norm_e_cg(1:cg_iter), 'k')
    semilogy(t_cgr(1:cgr_iter), norm_e_cgr(1:cgr_iter), 'b')
    
    legend('t\_sd', 't\_cg', 't\_cgr')
  end
  
  display('paused')
  pause
end % end of all the tests
