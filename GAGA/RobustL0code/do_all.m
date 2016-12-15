% DO_ALL.M ---- Numerical experiments for "Expander l_0-Decoding"
% Figure labels based on paper's first version available at http://arxiv.org/abs/1508.01256
% Rodrigo Mendoza-Smith, August 2015
%
% SETUP:
% 0. Compile gaga_smv 
% 1. Edit SET_HOME.m to the path of GAGA directory
% 2. Include results_X_S_smv.mat in numerics/phase_transitions/data/non_ccs folder
%    for X in {ALPS, CGIHT, CGIHTprojected, CGIHTrestarted, CSMPSP, FIHT, HTP, NIHT}
% 3. Include poytope.mat in numerics/phase_transitions/data/polytope folder


% Figure 2: 50% recovery probability logistic regression curves for \expander and n=2^{18}
% Figure 3: Average recovery time (ms) of fastest algorithm at each (\delta, \rho) for \expander and n=2^{18}
% Figure 4: Selection map of fastest algorithm at each (\delta, \rho) for \expander and n=2^{18}

run('phase_transitions/data/robust_l0/query.m');
run('phase_transitions/data/deterministic_robust_l0/query.m');
run('phase_transitions/data/ssmp/query.m');

run('phase_transitions/plot_data/plot_all.m');
fprintf('Plots in ./phase_transitions/plots/all/');

if false

    % Figure 5: Average recovery time (sec) with dependence on \rho for \delta = 0.01 and \expander with n = 2^{20}
    % Figure 6: Average recovery time (sec) with dependence on \rho for \delta = 0.1 and \expander with n = 2^{20}

    run('timing_fixed_delta/er/query.m');
    run('timing_fixed_delta/parallel_l0/query.m');
    run('timing_fixed_delta/parallel_lddsr/query.m');
    run('timing_fixed_delta/serial_l0/query.m');
    run('timing_fixed_delta/smp/query.m');
    run('timing_fixed_delta/ssmp/query.m');

    run('timing_fixed_delta/plot_timing_fixed_delta.m');
    fprintf('Plots in ./timing_fixed_delta/plots/');

    % Figure 7: Number of iterations to convergence for Parallel-l0, Serial-l0, parallel-LDDSR, at \delta=0.1 with \expander and n=2^{20}
    % Figure 8: Number of iterations to convergence for ER and SSMP, at \delta=0.1 with \expander and n=2^{20}

    run('number_of_iterations/number_of_iterations.m');
    fprintf('Plots in ./number_of_iterations/plots/');

    % Figure 9: Average recovery time (sec) for Parallel-l0 with dependence on \rho for \delta=0.001 and \expander with n\in\{ 2^{22}, 2^{24}, 2^{26}\}
    % Table III: Average recovery time (sec) for Parallel-l0 at \rho=0.05 and \delta=0.001 for n\in\{ 2^{22}, 2^{24}, 2^{26}\}

    run('low_delta_logistic/parallel_l0/query.m');

    run('low_delta_logistic/plot_timing_fixed_delta_single_50percent.m');
    run('low_delta_logistic/create_table_times_at_rho.m');
    fprintf('Plots in ./low_delta_logistic/plots/');

    % Figure 10: 50% recovery probability logistic regression curves for Parallel-l0 with \expander and n=2^{18}, with signals having a fixed proportion, {\em band}, or identical nonzero elements in its support.

    run('banded_signal/p_0/query.m');
    run('banded_signal/p_5/query.m');
    run('banded_signal/p_10/query.m');
    run('banded_signal/p_15/query.m');
    run('banded_signal/p_20/query.m');
    run('banded_signal/p_25/query.m');
    run('banded_signal/p_30/query.m');
    run('banded_signal/p_40/query.m');
    run('banded_signal/p_50/query.m');
    run('banded_signal/p_60/query.m');
    run('banded_signal/p_70/query.m');
    run('banded_signal/p_90/query.m');

    run('banded_signal/make_joint_transition_plots_by_band.m');
    fprintf('Plots in ./phase_transitions/plots/');

    % Figure 11: 50% recovery probability logistic regression curves for Parallel-l0 with \expander and n=2^{18} for d \in \{5, 7, 9 , ..., 17, 19\}.

    run('phase_transitions/data/parallel_l0/query_2.m');
    run('phase_transitions/data/parallel_l0/reprocess.m');

    run('phase_transitions/plot_data/plot_parallel_l0_several_d.m');
    fprintf('Plots in ./phase_transitions/plots/ccs/');

end
