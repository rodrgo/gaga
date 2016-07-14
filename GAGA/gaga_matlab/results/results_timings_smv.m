function [errors, timings, checkSupport, convRate] = results_timings_smv(vec, vec_input, vecDistribution, residNorm_prev, iter, checkSupport, totaltime, time_sum, time_per_iteration, time_supp_set, ksum, alpha, supp_flag, k, m, n, p, matrixEnsemble, seed, timeTest, algstr)
% results function




% compute the convergence rate
temp=max(1,min(iter,16));
convRate=(residNorm_prev(16)/residNorm_prev(17-temp))^(1/temp);

% identify the support set detection
checkSupport = check_support(vec_input,vec);

% compute the errors of the approximation versus the original vector
vec_input = vec_input - vec;
errors = [norm(vec_input, 1); norm(vec_input, 2); norm(vec_input, inf)];

% record the times in the order algorithm time, average iteration time,
% total time including problem generation and error computation.
% The timings are scaled to return their value in milliseconds.
timings = 1000*[totaltime time_sum/iter toc(timeTest)];

time_per_iteration = 1000*time_per_iteration;
time_supp_set = 1000*time_supp_set;

% write all data to text file
c = clock;
year=c(1);
month=c(2);
day=c(3);
fname=sprintf('matlab_timings_data_smv%d%02d%02d.txt',year,month,day);

foutput = fopen(fname, 'a+');
if (foutput < 3)
    display('WARNING: Output file did not open!')
else
    File_output_smv_timings(foutput, k, m, n, vecDistribution, errors, timings, iter, convRate, checkSupport, seed, time_per_iteration, time_supp_set, algstr, alpha, supp_flag, p, matrixEnsemble);
end
fclose(foutput);
