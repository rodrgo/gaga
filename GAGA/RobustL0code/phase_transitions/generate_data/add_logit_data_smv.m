function add_logit_data_smv(alg_list, tol)

ens_list = cell(1,1);
ens_list{1} = 'smv';

tic

for i=1:length(alg_list)
  for j=1:length(ens_list)
    add_logit_data(alg_list{i},ens_list{j}, tol);
    [i length(alg_list) j length(ens_list)]
    toc
  end
end

% end function
end
