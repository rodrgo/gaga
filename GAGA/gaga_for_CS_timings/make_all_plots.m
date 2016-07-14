
ttt=tic;
display('Making plots started');

make_supp_ratio_plot('NIHT','dct',2^10)
make_supp_ratio_plot('NIHT','dct',2^12)
% make_supp_ratio_plot('NIHT','dct',2^14)
% make_supp_ratio_plot('NIHT','dct',2^16)
% make_supp_ratio_plot('NIHT','dct',2^18)
% make_supp_ratio_plot('NIHT','dct',2^20)

display(sprintf('Plots completed in %f seconds', toc(ttt)));

