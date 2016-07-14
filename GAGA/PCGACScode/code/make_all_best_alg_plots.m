%turn on (1) or off (0) ensembles to be processed.
dct = 1;
smv = 1;
gen = 1;

% set the values of n for each matrix ensemble
n_list_dct=[2^10];
n_list_smv=[2^10];
n_list_gen=[2^10];
%n_list_dct=[2^16 2^18 2^20];
%n_list_smv=[2^14 2^16 2^18];
%n_list_gen=[2^10 2^12 2^14];

% set the algorithms
%{
alg_list=cell(5,1);
alg_list{1}='CGIHTrestarted';
alg_list{2}='CGIHT';
alg_list{3}='NIHT';
alg_list{4}='FIHT';
alg_list{5}='CSMPSP';
%alg_list{6}='HTP';
%}
alg_list=cell(3,1);
alg_list{1}='NIHT';
alg_list{2}='HTP';
alg_list{3}='CSMPSP';


nonzeros_list=[4 7];

noise_list = [0]
vec_list=[1];


for i=1:length(vec_list)
  for j=1:length(noise_list)

    if dct
      for q=1:length(n_list_dct)
        make_best_alg_plot('dct',n_list_dct(q),0,noise_list(j),vec_list(i),alg_list);
      end
    end

    if smv
      for q=1:length(n_list_smv)
        for p=1:length(nonzeros_list)
          make_best_alg_plot('smv',n_list_smv(q),nonzeros_list(p),noise_list(j),vec_list(i),alg_list);
        end
      end
    end

    if gen
      for q=1:length(n_list_gen)
        make_best_alg_plot('gen',n_list_gen(q),0,noise_list(j),vec_list(i),alg_list);
      end
    end

  end  % ends noise_list loop
end  % ends vec_list loop


