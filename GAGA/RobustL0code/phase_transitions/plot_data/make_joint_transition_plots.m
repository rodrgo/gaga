function make_joint_transition_plots(alg_list, ens, n, nonzeros, destination, simplex_curve, sigmas)
% ens should be 'dct', 'smv', or 'gen'
% for ens=='smv' the third (nonzeros) argument is required

if (strcmp(ens,'smv')==1 & nargin<3)
  warning('smv requires nonzeros to be specified')
end

assert(length(n) == 1);

% ================
% One plot per (noise_level,  n), one curve per algorithm
% ================

%clist='brkmgcybrk';
clist = createColorPalette(alg_list);

for ell = 1:length(sigmas)
	figure(ell)
	hold off

	% split by noise level
	for i=1:length(alg_list)
		fname=['results_' alg_list{i} '_S_' ens];
		load(fname)

		% Extract relevant data

		betas = results_cell{ell}.betas;
		deltas = results_cell{ell}.deltas;
		n_list = results_cell{ell}.n_list;
		nz_list = results_cell{ell}.nz_list;
		noise_level = results_cell{ell}.noise_level;

		% Extract relevant data
	
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

	fname1=[destination '/transition_all_' ens '_n_' num2str(n) '_sigma_' sprintf('%1.3f', noise_level)];
	if strcmp(ens,'smv')
		fname1=[fname1 '_d_' num2str(nonzeros)];
	end

	if strcmp(ens,'smv')
		txt=['50% phase transition curves for d = ' ...
			num2str(nonzeros) ' with n = 2^{' num2str(log2(n)) '} and \sigma = ' sprintf('%1.3f', noise_level)];
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

end



%end function
end
