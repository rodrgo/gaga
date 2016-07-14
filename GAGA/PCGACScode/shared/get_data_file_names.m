function fnames=get_data_file_names(ens,noise_level,no_zero_string,startdate,enddate)
% fnames=get_data_file_names(ens,noise_level,no_zero_string,startdate,enddate)
% This function returns a cell of file names for the data (.txt) files
% generated for the input arguments
%   ens is a string denoting a matrix ensemble from 'dct','gen','smv'
%   noise_level is a scalar (optional)
%   no_zero_string is a flag (optional)
%     if no_zero_string = 0, the noise-free data files should end with '_noise0.000.txt'  (Noise data for GAGA 1.0.0.)
%     if no_zero_string = 1, then no such ending string should be on the files.  (This is default for GAGA 1.1.0.)
%   startdate is a string 'yyyymmdd' (optional)
%   enddate is a string 'yyyymmdd' (optional)
% startdate and enddate provide a desired range of dates for the files

% This flag determines if specific dates were entered
specific_dates = 1;

if ( nargin==5 ) && ( str2num(startdate)>str2num(enddate) )
  error('The end date is earlier than the start date.');
end

if nargin<5
  if nargin == 4
    enddate = datestr(now,'yyyymmdd');  % if startdate is provided, but no enddate, the enddate is set to the current date
  elseif nargin < 4
    specific_dates = 0;                 % if no dates are given, the full set of files is returned
    if nargin<3
      no_zero_string = 1;               
      if nargin<2
        noise_level = 0;
      end
    end
  end
end


noise_string = ['_noise' num2str(noise_level)];
% The noise string must be 5 characters x.xxx so we append zeros as
% necessary.
switch length(num2str(noise_level))
  case 1
    noise_string = [noise_string '.' num2str(0) num2str(0) num2str(0)];
  case 2
    error('The noise_levels must be either an integer or have between one and three decimal places.')
  case 3
    noise_string = [noise_string num2str(0) num2str(0)];
  case 4
    noise_string = [noise_string num2str(0)];
  otherwise 
    error('The noise_levels must be either an integer or have between one and three decimal places.')
end

if (no_zero_string) && (noise_level==0)
  noise_string='';
end

fname_generic = sprintf('gpu_data_%s*%s.txt',ens,noise_string);
fname_length = length(fname_generic)+7;  % the * character represents 8 digit dates yyyymmdd
fnames_all=ls(fname_generic);

ind = strfind(fnames_all,'gpu_data');   % finds the begining of the appropriate file names 

% if the noise string should not be present, remove all files ending in '_noiseX.XXX.txt'
if (noise_level==0 && no_zero_string)
  remove_ind = strfind(fnames_all,'_noise');
  remove_ind = remove_ind - 20;  % the string '_noise' would appear 20 indices after the start 'gpu_data_matYYYYMMDD'
  ind = setdiff(ind,remove_ind);
end

fnames=cell(1,length(ind));

for j=1:length(ind)
  first=ind(j);
  last=ind(j)+fname_length-1;
  fnames{j}=fnames_all(first:last);
end

if specific_dates
  startdate_num = str2num(startdate)
  enddate_num = str2num(enddate)
  final_ind=[];
  for j=1:length(fnames)
    date_str=fnames{j}(13:20);
    tmp = str2num(date_str);
    if tmp >= startdate_num 
      if tmp <= enddate_num
        final_ind = [final_ind j];
      end
    end
  end
  fnames_tmp = fnames;
  fnames = cell(1,length(final_ind));
  for jj=1:length(final_ind)
    fnames{jj}=fnames_tmp{final_ind(jj)};
  end
end   


