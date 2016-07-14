function make_joint_transition_plots(alg_list, ens, n, nonzeros, destination, simplex_curve)
% ens should be 'dct', 'smv', or 'gen'
% for ens=='smv' the third (nonzeros) argument is required

name_split = strsplit(destination, '/');
tag = name_split{end};

if (strcmp(ens,'smv')==1 & nargin<3)
  warning('smv requires nonzeros to be specified')
end

%clist='brkmgcybrk';
clist = createColorPalette(alg_list);

figure(1)
hold off

for i=1:length(alg_list)
  fname=['results_' alg_list{i} '_S_' ens];
  load(fname)
	
	ind_n=find(n_list==n);
	if strcmp(ens,'smv')
		ind_nz=find(nz_list==nonzeros);
		ind_n=intersect(ind_n,ind_nz);
	end
	
	plot(deltas{ind_n},1./betas{ind_n}(:,2),'Color',clist(i,:));
	ll=ceil(length(deltas{ind_n})*(i-1/2)/length(alg_list));
	txt=['\leftarrow ' regexprep(alg_list{i},'_','\\_')];
	h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2),txt,'HorizontalAlignment','left');
	set(h,'color',clist(i,:))
	hold on
end

if simplex_curve
	i  = length(alg_list) + 1;
	fname=['polytope'];
	load(fname)
	polytope_data = rhoW_crosspolytope;
	plot(polytope_data(:,1),polytope_data(:,2),'Color',clist(i,:));
	ll=ceil(length(polytope_data(:,1))*(i-1/2)/(length(alg_list) + 1)); % position depends on previous plots
	txt=['\leftarrow ' regexprep('l1-regularization','_','\\_')];
	h=text(polytope_data(ll, 1),polytope_data(ll,2),txt,'HorizontalAlignment','left');
	set(h,'color',clist(i,:))
	hold on
end


fname1=[destination '/transition_all_' ens];
if strcmp(ens,'smv')
	fname1=[fname1 '_d_' num2str(nonzeros)];
end

if strcmp(ens,'smv')
	txt=['50% phase transition curve for d = ' ...
		num2str(nonzeros) ' with n = 2^{' num2str(log2(n)) '}'];
end

if strcmp(ens,'smv')
  title(strrep(txt, '_', '\_'),'fontsize',12)
else
	title(txt,'fontsize',14)
end

xlabel('\delta=m/n','fontsize',14);
ylabel('\rho=k/m','fontsize',14);
print(fname1,'-dpdf')
print('-depsc', strcat(fname1, '.eps'))

%end function
end
