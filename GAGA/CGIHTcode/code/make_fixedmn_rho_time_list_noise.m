% make the plot of time for fixed [m, n]

function [k_list, ave_time_list] = make_fixedmn_rho_time_list_noise(alg, ens, m, n, succ_prob, noise_level, nonzeros)
tol = 10^(-2)+2*noise_level;


noise_string = sprintf('_noise%0.3f',noise_level);
if (noise_level == 0)
  noise_string = '';
end

fname_load=['results_' alg '_S_' ens '_vecDistr1' noise_string '.mat'];
if exist(fname_load,'file') == 2
  load(fname_load);
else
  k_list = []; ave_time_list = [];
  return
end


if strcmp(ens,'gen')
  add_factor = 0;
  ind_search = find(results(:,2) == m & results(:,3) == n);
elseif strcmp(ens,'dct')
  add_factor = -1;
  ind_search = find(results(:,2) == m & results(:,3) == n);
elseif strcmp(ens, 'smv')
  add_factor = 1;
  ind_search = find(results(:,2) == m & results(:,3) == n & results(:,5) == nonzeros);
end

results = results(ind_search,:);

k_list = intersect(results(:,1),results(:,1));
k_list = sort(k_list);
ave_time_list = zeros(size(k_list));

for jj = 1 : length(k_list)
  ind = find(results(:,1)==k_list(jj));
  error_list = results(ind,add_factor+7); % l-two error
  time_list = results(ind,add_factor+9); % time
  if (sum(error_list < tol)/length(ind)) >= succ_prob 
    ave_time_list(jj) = (error_list < tol)'*time_list/sum(error_list < tol); 
  else
    break
  end 
end

k_list = k_list(1:jj-1);
ave_time_list = ave_time_list(1:jj-1);


