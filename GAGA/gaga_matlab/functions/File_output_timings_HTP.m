function []=File_output_timings_HTP(foutput, k, m, n, vecDistribution, errors, timings, iter, conv_rate, checkSupport, seed, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, algstr, alpha, supp_flag)


fprintf(foutput,'%s_M_dct output: ',algstr);
fprintf(foutput,'k %d, m %d, n %d, vecDistribution %d ',k,m,n,vecDistribution);
fprintf(foutput,'errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ',errors(1),errors(2),errors(3));
fprintf(foutput,'timeALG %0.7e timeIter %0.7e timeTotal %0.7e ',timings(1),timings(2),timings(3));
fprintf(foutput,'iterations %d ',iter);
fprintf(foutput,'convergence_rate %0.7e ',conv_rate);
fprintf(foutput,'TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ',checkSupport(1),checkSupport(2),checkSupport(3),checkSupport(4));
fprintf(foutput,'seed %d ',seed);
fprintf(foutput,'alpha %0.7e ',alpha);
fprintf(foutput,'supp_flag %d \n',supp_flag);

for j = 1:iter
    fprintf(foutput, 'iteration %d time_iteration %0.7e time_supp_set %0.7e, cg_per_iteration %0.0f, time_for_cg %0.7e\n',j, time_per_iteration(j), time_supp_set(j), cg_per_iteration(j), time_for_cg(j));
end


