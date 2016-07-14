function make_combined_table(alg,matrixEnsemble,supp_flag,tableType,sds_to_add)


if (strcmp(alg,'NIHT') | strcmp(alg,'HTP')) & strcmp(matrixEnsemble,'dct')

  [rows_niht_gpu choice_niht_gpu]=make_table_reduced_rows(alg,'gpu',matrixEnsemble,supp_flag,tableType,sds_to_add);
  [rows_niht_cpu choice_niht_cpu]=make_table_reduced_rows(alg,'matlab',matrixEnsemble,supp_flag,tableType,sds_to_add);

  row_breaks=[1 length(rows_niht_gpu)];
  row_breaks=[row_breaks; sum(row_breaks(end,:)) length(rows_niht_cpu)];

  row_break_text=cell(2,1);
  row_break_text{1}='gpu';
  row_break_text{2}='cpu';

  choice_combined=[choice_niht_gpu; choice_niht_cpu];
  table_reduced_choice=intersect(choice_combined,choice_combined);
  table_reduced=[rows_niht_gpu; rows_niht_cpu];

elseif (strcmp(alg,'NIHT') | strcmp(alg,'HTP')) & strcmp(matrixEnsemble,'gen')

  [rows_niht_gpu choice_niht_gpu]=make_table_reduced_rows(alg,'gpu',matrixEnsemble,supp_flag,tableType,sds_to_add);
  [rows_niht_cpu choice_niht_cpu]=make_table_reduced_rows(alg,'matlab',matrixEnsemble,supp_flag,tableType,sds_to_add);

  row_breaks=[1 length(rows_niht_gpu)];
  row_breaks=[row_breaks; sum(row_breaks(end,:)) length(rows_niht_cpu)];

  row_break_text=cell(2,1);
  row_break_text{1}='gpu';
  row_break_text{2}='cpu';

  choice_combined=[choice_niht_gpu; choice_niht_cpu];
  table_reduced_choice=intersect(choice_combined,choice_combined);
  table_reduced=[rows_niht_gpu; rows_niht_cpu];

elseif (strcmp(alg,'NIHT') | strcmp(alg,'HTP')) & strcmp(matrixEnsemble,'smv')

  [rows_niht_gpu4 choice_niht_gpu4]=make_table_reduced_rows(alg,'gpu',matrixEnsemble,supp_flag,tableType,sds_to_add,4);
  [rows_niht_cpu4 choice_niht_cpu4]=make_table_reduced_rows(alg,'matlab',matrixEnsemble,supp_flag,tableType,sds_to_add,4);
  [rows_niht_gpu7 choice_niht_gpu7]=make_table_reduced_rows(alg,'gpu',matrixEnsemble,supp_flag,tableType,sds_to_add,7);
  [rows_niht_cpu7 choice_niht_cpu7]=make_table_reduced_rows(alg,'matlab',matrixEnsemble,supp_flag,tableType,sds_to_add,7);
  [rows_niht_gpu13 choice_niht_gpu13]=make_table_reduced_rows(alg,'gpu',matrixEnsemble,supp_flag,tableType,sds_to_add,13);
  [rows_niht_cpu13 choice_niht_cpu13]=make_table_reduced_rows(alg,'matlab',matrixEnsemble,supp_flag,tableType,sds_to_add,13);
  
  row_breaks=[1 length(rows_niht_gpu4)];
  row_breaks=[row_breaks; sum(row_breaks(end,:)) length(rows_niht_cpu4)];
  row_breaks=[row_breaks; sum(row_breaks(end,:)) length(rows_niht_gpu7)];
  row_breaks=[row_breaks; sum(row_breaks(end,:)) length(rows_niht_cpu7)];
  row_breaks=[row_breaks; sum(row_breaks(end,:)) length(rows_niht_gpu13)];
  row_breaks=[row_breaks; sum(row_breaks(end,:)) length(rows_niht_cpu13)];
  
  row_break_text=cell(6,1);
  row_break_text{1}='gpu p=4';
  row_break_text{2}='cpu p=4';
  row_break_text{3}='gpu p=7';
  row_break_text{4}='cpu p=7';
  row_break_text{5}='gpu p=13';
  row_break_text{6}='cpu p=13';


  choice_combined=[choice_niht_gpu4; choice_niht_cpu4; choice_niht_gpu7; ...
                  choice_niht_cpu7; choice_niht_gpu13; choice_niht_cpu13]; 
  table_reduced_choice=intersect(choice_combined,choice_combined);
  table_reduced=[rows_niht_gpu4; rows_niht_cpu4; rows_niht_gpu7; ...
                 rows_niht_cpu7; rows_niht_gpu13; rows_niht_cpu13]; 
   
end


%make the file output name for the table
fname_reduced='tables/';
if tableType==1
  fname_reduced=[fname_reduced 'table_time_per_iteration_'];
elseif tableType==2
  fname_reduced=[fname_reduced 'table_time_supp_set_'];
elseif tableType==3
  fname_reduced=[fname_reduced 'table_time_cg_per_iteration_'];
elseif tableType==4
  fname_reduced=[fname_reduced 'table_total_cg_steps_'];
elseif tableType==5
  fname_reduced=[fname_reduced 'table_time_problem_generation_'];
end

fname_reduced=[fname_reduced sprintf('combined_%s_%s_supp_flag_%d_sds_%d.tex',alg,matrixEnsemble,supp_flag,round(sds_to_add))];

fid_reduced=fopen(fname_reduced,'wt');



% begin information for table
tmp=sprintf('\\\\begin{table}[h]\n');
fprintf(fid_reduced,tmp);

%make the caption
model=cell(4,1);
model{1}='$Const.$ ';
model{2}='$Const. + \\alpha\\delta$ ';
model{3}='$Const. + \\beta\\rho$ ';
model{4}='$Const. + \\alpha\\delta + \\beta\\rho$ ';

cap='Least squares model ';

if (ismember(4,table_reduced_choice) | (ismember(2,table_reduced_choice) ...
                                       & ismember(3,table_reduced_choice)))
  cap=[cap model{4}];
elseif ismember(3,table_reduced_choice)
  cap=[cap model{3}];
elseif ismember(2,table_reduced_choice)
  cap=[cap model{2}];
else 
  cap=[cap model{1}];
end
cap = [cap 'for the average time in \\underline{milliseconds} '];



if tableType==1
  cap=[cap 'of ' alg ' iteration lines 1 and 4 using the '];
  cap=[cap '${' matrixEnsemble '}$ matrix ensemble.  This time '];
  cap=[cap 'includes both a call to $\\rsd$ as well as '];
  cap=[cap 'updating the stopping criteria.'];
elseif tableType==3
  cap=[cap 'of the projection portion of $\\cgproject$ '];
  cap=[cap 'divided by the number of CG iterations '];
  cap=[cap 'conduced, to approximate the average time of '];
  cap=[cap '$\\rcg$, with values recorded from the interior '];
  cap=[cap 'of HTP iteration line 4 for the '];
  cap=[cap '${' matrixEnsemble '}$ matrix ensemble.'];
elseif tableType==5
  cap=[cap 'for random problem generation and memory '];
  cap=[cap 'allocation for ' alg ' using the '];
  cap=[cap '${' matrixEnsemble '}$ matrix ensemble.'];
end


%cap=['tables/caption_' fname_reduced(8:end)];





label=[alg '_' matrixEnsemble '_' num2str(tableType)];

%tmp=sprintf('\\\\caption[]{\\\\input{%s}}\\\\label{table:%s}\n',cap,label);
tmp=sprintf('\\\\caption[]{%s}\\\\label{table:%s}\n',cap,label);
fprintf(fid_reduced,tmp);

tmp=sprintf('\\\\begin{center}\n');
fprintf(fid_reduced,tmp);



%if table_reduced choice has only option 1 then we need to drop 
%both delta and rho.
if (ismember(1,table_reduced_choice) && length(table_reduced_choice)==1)
  for j=1:length(table_reduced)
    i1=findstr(table_reduced{j},'- & ');
    i2=[1:1:i1-1 i1+4:1:length(table_reduced{j})];
    table_reduced{j}=table_reduced{j}(i2);
  end
  for j=1:length(table_reduced)
    i1=findstr(table_reduced{j},'- & ');
    i2=[1:1:i1-1 i1+4:1:length(table_reduced{j})];
    table_reduced{j}=table_reduced{j}(i2);
  end
  tmp=sprintf('\\\\begin{tabular}{|l|l|l|l|}\n');
  fprintf(fid_reduced,tmp);
    
  tmp=sprintf('\\\\hline\n');
  fprintf(fid_reduced,tmp);
  tmp=sprintf('& n & Const. & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
  fprintf(fid_reduced,tmp);  
  %if table_reduced choice has only option 2 or 3 then we need to
  %drop the one variable not used.
elseif ismember(4,table_reduced_choice)==0 && ( (ismember(2,table_reduced_choice)==0) || (ismember(3,table_reduced_choice)==0) )
    for j=1:length(table_reduced)
      i1=findstr(table_reduced{j},'- & ');
      i2=[1:1:i1-1 i1+4:1:length(table_reduced{j})];
      table_reduced{j}=table_reduced{j}(i2);
    end
    if ismember(2,table_reduced_choice)==1
      tmp=sprintf('\\\\begin{tabular}{|l|l|l|l|l|}\n');
      fprintf(fid_reduced,tmp);
      
      tmp=sprintf('\\\\hline\n');
      fprintf(fid_reduced,tmp);
      tmp=sprintf('& n & Const. & $\\\\alpha$ & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
      fprintf(fid_reduced,tmp);
    else
      tmp=sprintf('\\\\begin{tabular}{|l|l|l|l|l|}\n');
      fprintf(fid_reduced,tmp);
      
      tmp=sprintf('\\\\hline\n');
      fprintf(fid_reduced,tmp);
      tmp=sprintf('& n & Const. & $\\\\beta$ & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
      fprintf(fid_reduced,tmp);
    end
else
  
  tmp=sprintf('\\\\begin{tabular}{|l|l|l|l|l|l|}\n');
  fprintf(fid_reduced,tmp);
  
  tmp=sprintf('\\\\hline\n');
  fprintf(fid_reduced,tmp);
  tmp=sprintf('& n & Const. & $\\\\alpha$ & $\\\\beta$ & $\\\\ell_{\\\\infty}$ error \\\\\\\\\n\\\\hline\\\\hline\n');
  fprintf(fid_reduced,tmp);
end

c=intersect(choice_combined,choice_combined);
if max(c)==1
  cline_offset=4;
elseif max(c)==4 | (ismember(2,c) & ismember(3,c))
  cline_offset=6;
else
  cline_offset=5;
end


%now print the data in table_reduced
%warning('off','all');
for zz=1:length(table_reduced)
  [insert_row ind]=ismember(zz,row_breaks(:,1));
  if (insert_row==1 & zz~=1)
    fprintf(fid_reduced,'\\hline');
  else
    tmp=sprintf('\\\\cline{2-%d}',cline_offset);
    fprintf(fid_reduced,tmp);
  end
  fprintf(fid_reduced,'\n');
  if insert_row==1
    tmp=sprintf('\\\\multirow{%d}{*}{%s}',row_breaks(ind,2),row_break_text{ind});
    fprintf(fid_reduced,tmp);
  end
  
  fprintf(fid_reduced,' & ');    
  fprintf(fid_reduced,table_reduced{zz});
  fprintf(fid_reduced,'\n');
end
fprintf(fid_reduced,'\\hline');
fprintf(fid_reduced,'\n');
%warning('on','all');


tmp=sprintf('\\\\end{tabular}\n');
fprintf(fid_reduced,tmp);
tmp=sprintf('\\\\end{center}\n');
fprintf(fid_reduced,tmp);
tmp=sprintf('\\\\end{table}\n');
fprintf(fid_reduced,tmp);

fclose(fid_reduced);





