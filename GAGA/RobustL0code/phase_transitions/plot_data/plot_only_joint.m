
clear all;

set(gca,'fontsize',14)

ns = [2^14, 2^16, 2^18];
%ns = [2^14];
nzs = [7];
ens_list = {'smv'};
sigmas = [1e-3, 1e-2, 1e-1];
SOL_TOL = 1;

simplex_curve = 0;
addpath('../data/polytope/');

alg_list = {'robust_l0', 'robust_l0_adaptive', 'robust_l0_trans', 'robust_l0_adaptive_trans', 'ssmp_robust', 'smp_robust', 'cgiht_robust'};

for i = 1:length(alg_list)
	dir = fullfile('../data/', alg_list{i});
	addpath(dir);
end

destination = fullfile('../plots/');

% 50% phase transitions with all n on one plot.


% Plot probability transforms for robust-l0-trans

%plot_probability_transforms;

% 50% phase transitions for given (ens, n, nonzeros)
% all algorithms shown on one plot.

if true 

	% make_joint_transition_plots

	for i = 1:length(ns)
		for j = 1:length(nzs)
			make_joint_transition_plots(alg_list, 'smv', ns(i), nzs(j), destination, simplex_curve, sigmas);
		end
	end

end


