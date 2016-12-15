function process_smv_data(alg_list, vecDistribution)

% Get filenames starting with 'gpu_data_smv'

files = strsplit(ls);

fnames = {};
fdates = {};
fname_start='gpu_data_smv';

for i = 1:length(files)
    if ~isempty(strfind(files{i}, fname_start))

        % Extract file name

        [~, fname, fext] = fileparts(files{i});
        fnames{end + 1} = [fname, fext];

        % Extract date string (as text) from file name

        tag_position = strfind(fname, '_noise');
        if ~isempty(tag_position) 
            fdates{end + 1} = fname(length(fname_start) + 1:(tag_position-1));
        else
            fdates{end + 1} = fname(length(fname_start) + 1:end);
        end

    end
end

fdatenums = cellfun(@str2num, fdates(:));

% Sort according to fdatenums

[~, idx] = sort(fdatenums);
fdatenums = fdatenums(idx);
fnames = fnames(idx);

% Read data

startdate = num2str(min(fdatenums));

results=[];
results_long=[];
results_full=[];
for i = 1:length(fnames)

    date_string = num2str(fdatenums(i));
    fname = fnames{i};
	fprintf('Processing %s\n', fname);
    %fname=[fname_start date_string '.txt'];

    fid = fopen(fname);
    if fid~=-1
        results_long = textscan(fid,'%s %s %s %f %s %s %f %s %s %f %s %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f %s %f');
        fclose(fid);

        if isempty(results_full) %strcmp(date_string,startdate)==1
            results_full=results_long;
        else
            for j=1:size(results_long,2)
                results_full{j}=[results_full{j}; results_long{j}];
            end
        end
        clear results_long
    end

end

%the data stored in results is:
%1, algorithm
%4, k
%7, m
%10, n
%13, vecDistribution
%15, nonzeros_per_column
%17, matrixEnsemble
%19, error in l1 norm
%21, error in l2 norm
%23, error in l_infinity norm
%25, time for the algorithm
%27, time per iteration
%29, total time
%31, iterations
%33, convergence rate
%35, number of true positive 
%37, number of false positive
%39, number of true negative
%41, number of flase negative
%43, random number generator seed
%45, band_percentage 
%47, noise_level 

columns_used=[4 7 10 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47];
columns='values in the columns of results are: k ,m, n,vecDistribution,nonzeros per column, matrix ensemble, l_one error,l_two error,l_infinity error, time for the algorithm, time per iteration,time total, iterations, convergence rate, number of true positive, number of false positives,number of true negatives, number of false negatives, random number genertor, seed, band_percentage, noise_level';

% Add results separated by noise level
% We want to create results first

noise_levels = unique(results_full{end});

for j=1:length(alg_list)

  % Get indices for algorithm alg_list{j}

  alg_ind = ismember(results_full{1}, alg_list{j});

  % Create a results array for this algorithm

  results_cell = {};

  for i = 1:length(noise_levels)

    % Get indices for this noise level

    noise_level = noise_levels(i);
    noise_ind = ismember(results_full{end}, noise_level);

    % Combine indices

    ind = alg_ind & noise_ind;

    if any(ind)

      results = zeros(nnz(ind),length(columns_used));
      for pp=1:length(columns_used)
        results(:,pp) = results_full{columns_used(pp)}(ind);
      end

      results_struct = [];
      results_struct.results = results;
      results_struct.columns = columns;
      results_struct.noise_level = noise_level;

      results_cell{end + 1} = results_struct;

    end

  end

  fname_save=['results_' alg_list{j} '.mat'];
  save(fname_save, 'results_cell');

end

% end function
end
