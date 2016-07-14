function []=File_output(foutput, k, m, n, vecDistribution, errors, timings, iter, conv_rate, checkSupport, seed, algstr)

fprintf(foutput,'%s_M_dct output: ',algstr);
fprintf(foutput,'k %d, m %d, n %d, vecDistribution %d ',k,m,n,vecDistribution);
fprintf(foutput,'errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ',errors(1),errors(2),errors(3));
fprintf(foutput,'timeALG %0.7e timeIter %0.7e timeTotal %0.7e ',timings(1),timings(2),timings(3));
fprintf(foutput,'iterations %d ',iter);
fprintf(foutput,'convergence_rate %0.7e ',conv_rate);
fprintf(foutput,'TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ',checkSupport(1),checkSupport(2),checkSupport(3),checkSupport(4));
fprintf(foutput,'seed %d \n',seed);

