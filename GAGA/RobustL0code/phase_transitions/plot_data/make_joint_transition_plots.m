function make_joint_transition_plots(alg_list, ens, n, nonzeros, destination, simplex_curve, sigmas, name_tag)
% ens should be 'dct', 'smv', or 'gen'
% for ens=='smv' the third (nonzeros) argument is required

% Font size
fs = [];
fs.title = 20;
fs.legend = 17;
fs.axis = 20;
fs.ticks = 20;

MS = 'MarkerSize';
LW = 'LineWidth';

if nargin < 8
	name_tag = '';
end

if (strcmp(ens,'smv')==1 & nargin<3)
  warning('smv requires nonzeros to be specified')
end

assert(length(n) == 1);

% ================
% One plot per (noise_level,  n), one curve per algorithm
% ================

%clist='brkmgcybrk';
clist = createColorPalette(alg_list);

s_list = '+o*xsdh^v><p.';

for ell = 1:length(sigmas)
	figure(ell)
	set(gcf, 'color', [1 1 1]);
	set(gca, 'Fontname', 'Times', 'Fontsize', 15);
	hold off

	handles = [];
	labels = {};

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
		
		handles(end + 1) = plot(deltas{ind_n},1./betas{ind_n}(:,2), strcat('-', s_list(i)), 'Color',clist(i,:), MS, 10);
		labels{end + 1} = regexprep(change_names(alg_list{i}),'_','\\_');
		ll=ceil(length(deltas{ind_n})*(i-1/2)/length(alg_list));
		txt=['\leftarrow ' regexprep(change_names(alg_list{i}),'_','\\_')];
		if deltas{ind_n}(ll) > 0.9
			ll = ll - 5
		end
		if false
			h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2),txt,'HorizontalAlignment','left', 'FontSize', fs.legend);
			set(h,'color',clist(i,:))
		end
		hold on

	end

	legend(handles, labels, 'Location', 'NorthWest', 'FontSize', fs.legend);
	
	if simplex_curve
		i  = length(alg_list) + 1;
		fname=['polytope'];
		load(fname)
		polytope_data = rhoW_crosspolytope;
		plot(polytope_data(:,1),polytope_data(:,2),'Color',clist(i,:));
		ll=ceil(length(polytope_data(:,1))*(i-1/2)/(length(alg_list) + 1)); % position depends on previous plots
		txt=['\leftarrow ' regexprep('l1-regularization','_','\\_')];
		h=text(polytope_data(ll, 1),polytope_data(ll,2),txt,'HorizontalAlignment','left', 'FontSize', fs.legend);
		set(h,'color',clist(i,:))
		hold on
	end

	fname1=[destination '/transition_all_' ens '_n_' num2str(n) '_sigma_' sprintf('%1.3f', noise_level)];
	if strcmp(ens,'smv')
		if isempty(name_tag)
			fname1=[fname1 '_d_' num2str(nonzeros)];
		else
			fname1=[fname1 '_d_' num2str(nonzeros) '_' name_tag];
		end
	end

	if strcmp(ens,'smv')
		title_str = {'50 percent phase transition', sprintf('$$d = %i$$, $$n = 2^{%i}$$, $$\\sigma = %1.3f$$', nonzeros, log2(n), noise_level)};
		%txt=['50% phase transition curves for d = ' ...
		%	num2str(nonzeros) ' with n = 2^{' sprintf('%d', num2str(log2(n))) '} and \sigma = ' sprintf('%1.3f', noise_level)];
	end

	if strcmp(ens,'smv')
		title(title_str, 'interpreter', 'latex', 'FontSize', fs.title);
		%title(strrep(txt, '_', '\_'),'fontsize',12)
	else
		title(title_str, 'interpreter', 'latex');
		%title(txt,'fontsize',14)
	end

	ylim([0 0.7]);
	xlabel('\delta=m/n','FontSize',fs.axis);
	ylabel('\rho=k/m','FontSize',fs.axis);

	% Tick size
	xt = get(gca, 'XTick');
	set(gca, 'FontSize', fs.ticks);

	xt = get(gca, 'YTick');
	set(gca, 'FontSize', fs.ticks);

	print(strcat(fname1, '.pdf'),'-dpdf')
	print('-depsc', strcat(fname1, '.eps'))

end



%end function
end
