function []=File_output_smv(foutput, k, m, n, vecDistribution, errors, timings, iter, conv_rate, checkSupport, seed, algstr, p, matrixEnsemble)

fprintf(foutput, '%s_M_smv output: ', algstr);
fprintf(foutput,'k %d, m %d, n %d, vecDistribution %d nonzeros_per_column %d matrixEnsemble %d ',k,m,n,vecDistribution,p,matrixEnsemble);
fprintf(foutput,'errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ',errors(1),errors(2),errors(3));
fprintf(foutput,'timeALG %0.7e timeIter %0.7e timeTotal %0.7e ',timings(1),timings(2),timings(3));
fprintf(foutput,'iterations %d ',iter);
fprintf(foutput,'convergence_rate %0.7e ',conv_rate);
fprintf(foutput,'TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ',checkSupport(1),checkSupport(2),checkSupport(3),checkSupport(4));
fprintf(foutput,'seed %d \n',seed);

