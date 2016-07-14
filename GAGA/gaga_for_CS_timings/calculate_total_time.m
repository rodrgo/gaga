fnames=ls;
fname_start='results_';
%look for files with the above starting string
ind=[];
for j=1:length(fnames)-length(fname_start)
  if strcmp(fname_start,fnames(j:j-1+length(fname_start)))
    ind=[ind j];
  end
end

%go through and load each mat file starting with fname_start and
%add the total time per problem to calculate the total time of all
%tests whose data is presented.

totalTime=0;

for j=1:length(ind)
  %find the break between file names
  k=ind(j);
  while(k<length(fnames))
    k=k+1;
    if strcmp(fnames(k-3:k),'.mat') 
      break
    end
  end
  load(fnames(ind(j):k));
  %add the times in the third column of results
  for n=1:length(results)
    totalTime=totalTime+sum(results{n}(:,3));
  end
end

display(sprintf('The total time is %d seconds, or %d days',totalTime/1000,totalTime/1000/3600/24))

