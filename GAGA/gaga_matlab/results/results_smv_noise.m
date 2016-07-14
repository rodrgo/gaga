function [errors, timings, checkSupport, convRate] = results_smv_noise(vec, vec_input, vecDistribution, residNorm_prev, iter, checkSupport, totaltime, time_sum, ksum, k, m, n, p, matrixEnsemble, seed, noise_level, timeTest, algstr)
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

% write all data to text file
c = clock;
year=c(1);
month=c(2);
day=c(3);
fname=sprintf('matlab_data_smv%d%02d%02d_noise%0.2f.txt',year,month,day,noise_level);

foutput = fopen(fname, 'a+');
if (foutput < 3)
    display('WARNING: Output file did not open!')
else
    File_output_smv(foutput, k, m, n, vecDistribution, errors, timings, iter, convRate, checkSupport, seed, algstr, p, matrixEnsemble);
end
fclose(foutput);
