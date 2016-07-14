function make_acceleration_table()

alg_list=cell(2,1);
alg_list{1}='NIHT';
alg_list{2}='HTP';

matrixEnsemble_list=cell(3,1);
matrixEnsemble_list{1}='dct';
matrixEnsemble_list{2}='smv';
matrixEnsemble_list{3}='gen';

nonZero_list=[4 7 13];

n_list=[2^10 2^12 2^14 2^16 2^18 2^20];
ensemble_n_max=zeros(3,1);
ensemble_n_max(1)=2^12;
ensemble_n_max(2)=2^12;
ensemble_n_max(3)=2^10;



%the table for NIHT will include 5 columns: matrixEnsemble, n,
%number of nonzeros if smv, descent acceleration, and support set 
%acceleration

fname_niht='tables/table_acceleration_niht.tex';

%the table for HTP will include 6 columns: matrixEnsemble, n,
%number of nonzeros if smv, descent acceleration, support set
%acceleration, and cg acceleration.

fname_htp='tables/table_acceleration_htp.tex';


fid_niht=fopen(fname_niht,'wt');
fid_htp=fopen(fname_htp,'wt');


cap_table=cell(2,1);
cap_table{1}='Multiplicative acceleration factor for NIHT of median times for the gpu over cpu times.';
cap_table{2}='Multiplicative acceleration factor for HTP of median times for the gpu over cpu times.';

label=cell(2,1);
label{1}='niht_accel';
label{2}='htp_accel';



% begin information for table
tmp=sprintf('\\\\begin{table}[h]\n');
fprintf(fid_niht,tmp);
fprintf(fid_htp,tmp);

tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\n',cap_table{1},label{1});
fprintf(fid_niht,tmp);

tmp=sprintf('\\\\caption{%s}\\\\label{table:%s}\n',cap_table{2},label{2});
fprintf(fid_htp,tmp);

tmp=sprintf('\\\\begin{center}\n');
fprintf(fid_niht,tmp);
fprintf(fid_htp,tmp);
  
tmp=sprintf('\\\\begin{tabular}{|l|l|c|l|l|l|}\n');
fprintf(fid_niht,tmp);

tmp=sprintf('\\\\begin{tabular}{|l|l|c|l|l|l|l|}\n');
fprintf(fid_htp,tmp);

tmp=sprintf('\\\\hline\n');
fprintf(fid_niht,tmp);
fprintf(fid_htp,tmp);



%first make the table for niht.

tmp=sprintf(' & n & p & $\\\\rsd$ & $\\\\fsupp$ & Prob.\\\\ Gen.\\\\ \\\\\\\\\n\\\\hline\\\\hline\n');
fprintf(fid_niht,tmp);  

%outer loop of matrixEnsemble, inner loop of n
for j=1:length(matrixEnsemble_list)
  if strcmp(matrixEnsemble_list{j},'smv')
    a=sum(n_list<=ensemble_n_max(j))*length(nonZero_list);
    tmp=sprintf('\\\\multirow{%d}{*}{%s}',a,'smv');
    fprintf(fid_niht,tmp);
    for q=1:length(nonZero_list)
      for k=1:length(n_list)
        if n_list(k)<=ensemble_n_max(j)
          accel_factors=calculate_cpu_gpu_acceleration('NIHT',matrixEnsemble_list{j},n_list(k),nonZero_list(q));
          tmp=sprintf(' & $2^{%d}$ & %d & %0.2f & %0.2f & %0.2f \\\\\\\\\n',log2(n_list(k)),nonZero_list(q),accel_factors(1),accel_factors(2),accel_factors(3));
          fprintf(fid_niht,tmp); 
          if q<length(nonZero_list) | n_list(k)<ensemble_n_max(j)
            tmp=sprintf('\\\\cline{2-6}\n');
            fprintf(fid_niht,tmp); 
          else
            tmp=sprintf('\\\\hline\n');
            fprintf(fid_niht,tmp);
          end
        end
      end
    end
  else
    a=sum(n_list<=ensemble_n_max(j));
    tmp=sprintf('\\\\multirow{%d}{*}{%s}',a,matrixEnsemble_list{j});
    fprintf(fid_niht,tmp);
    for k=1:length(n_list)
      if n_list(k)<=ensemble_n_max(j)
        accel_factors=calculate_cpu_gpu_acceleration('NIHT',matrixEnsemble_list{j},n_list(k));
        tmp=sprintf(' & $2^{%d}$ & & %0.2f & %0.2f & %0.2f \\\\\\\\\n',log2(n_list(k)),accel_factors(1),accel_factors(2),accel_factors(3));
        fprintf(fid_niht,tmp); 
        if n_list(k)<ensemble_n_max(j)
          tmp=sprintf('\\\\cline{2-6}\n');
          fprintf(fid_niht,tmp); 
        else
          tmp=sprintf('\\\\hline\n');
          fprintf(fid_niht,tmp);
        end
      end
    end
  end
end



%second make the table for htp.

tmp=sprintf(' & n & p & $\\\\rsd$ & $\\\\fsupp$ & Prob.\\\\ Gen.\\\\ & $\\\\rcg$ \\\\\\\\\n\\\\hline\\\\hline\n');
fprintf(fid_htp,tmp);  

%outer loop of matrixEnsemble, inner loop of n
for j=1:length(matrixEnsemble_list)
  if strcmp(matrixEnsemble_list{j},'smv')
    a=sum(n_list<=ensemble_n_max(j))*length(nonZero_list);
    tmp=sprintf('\\\\multirow{%d}{*}{%s}',a,'smv');
    fprintf(fid_htp,tmp);
    for q=1:length(nonZero_list)
      for k=1:length(n_list)
        if n_list(k)<=ensemble_n_max(j)
          accel_factors=calculate_cpu_gpu_acceleration('HTP',matrixEnsemble_list{j},n_list(k),nonZero_list(q));
          tmp=sprintf('& $2^{%d}$ & %d & %0.2f & %0.2f & %0.2f & %0.2f \\\\\\\\\n',log2(n_list(k)),nonZero_list(q),accel_factors(1),accel_factors(2),accel_factors(3),accel_factors(4));
          fprintf(fid_htp,tmp);
          if q<length(nonZero_list) | n_list(k)<ensemble_n_max(j)
            tmp=sprintf('\\\\cline{2-7}\n');
            fprintf(fid_htp,tmp); 
          else
            tmp=sprintf('\\\\hline\n');
            fprintf(fid_htp,tmp);
          end
        end
      end
    end
  else
    a=sum(n_list<=ensemble_n_max(j));
    tmp=sprintf('\\\\multirow{%d}{*}{%s}',a,matrixEnsemble_list{j});
    fprintf(fid_htp,tmp);
    for k=1:length(n_list)
      if n_list(k)<=ensemble_n_max(j)
        accel_factors=calculate_cpu_gpu_acceleration('HTP',matrixEnsemble_list{j},n_list(k));
        tmp=sprintf('& $2^{%d}$ & & %0.2f & %0.2f & %0.2f & %0.2f \\\\\\\\\n\\\\cline{2-7}\n',log2(n_list(k)),accel_factors(1),accel_factors(2),accel_factors(3),accel_factors(4));
        fprintf(fid_htp,tmp); 
        if n_list(k)<ensemble_n_max(j)
          tmp=sprintf('\\\\cline{2-7}\n');
          fprintf(fid_htp,tmp); 
        else
          tmp=sprintf('\\\\hline\n');
          fprintf(fid_htp,tmp);
        end
      end
    end
  end
end




tmp=sprintf('\\\\end{tabular}\n');
fprintf(fid_niht,tmp);
fprintf(fid_htp,tmp);
tmp=sprintf('\\\\end{center}\n');
fprintf(fid_niht,tmp);
fprintf(fid_htp,tmp);
tmp=sprintf('\\\\end{table}\n');
fprintf(fid_niht,tmp);
fprintf(fid_htp,tmp);

fclose(fid_niht);
fclose(fid_htp);



