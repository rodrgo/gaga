function make_transition_plots(alg_list, ens_list, destination)

% Font size
fs = [];
fs.title = 20;
fs.legend = 17;
fs.axis = 20;
fs.ticks = 20;

name_split = strsplit(destination, '/');
tag = name_split{end};

tic
format shortg

%clist='brkgmcybrk';
clist = createColorPalette(alg_list);

s_list = '+o*xsdh^v><p.';
MS = 'MarkerSize';
LW = 'LineWidth';

% ================
% One plot per algorithm, for all n, for all noise_levels
% ================

for j=1:length(ens_list)
	for i=1:length(alg_list)
		fname=['results_' alg_list{i} '_S_' ens_list{j}];
		load(fname)

		% Create plots

		figure(1)
		set(gcf, 'color', [1 1 1]);
		set(gca, 'Fontname', 'Times', 'Fontsize', 15);
		hold off

		handles1 = zeros(length(results_cell), 1);
		legends1 = cell(length(results_cell), 1);

		figure(3)
		set(gcf, 'color', [1 1 1]);
		set(gca, 'Fontname', 'Times', 'Fontsize', 15);
		hold off

		handles3 = zeros(length(results_cell), 1);
		legends3 = cell(length(results_cell), 1);

		% Spit by n

		n_list_global = unique(results_cell{1}.n_list);
		nz_list_global = unique(results_cell{1}.nz_list);
		for ell = 1:length(results_cell)
			assert(isequal(n_list_global, results_cell{ell}.n_list), 'n_list not equal');
			assert(isequal(nz_list_global, unique(results_cell{ell}.nz_list)), 'nz_list not equal');
		end

		assert(length(unique(nz_list_global)) == 1, 'More than one nz element. Not supported.');
		d = nz_list_global(1);

		for ii = 1:length(n_list_global)

			n = n_list_global(ii);

			for ell = 1:length(results_cell)

				% Extract relevant data

				betas = results_cell{ell}.betas;
				deltas = results_cell{ell}.deltas;
				n_list = results_cell{ell}.n_list;
				nz_list = results_cell{ell}.nz_list;
				noise_level = results_cell{ell}.noise_level;

				if strcmp(ens_list{j},'smv')

					% Get position

					pos = find(n_list == n);

					% Start plotting

					figure(1)

					r_star=1./betas{pos}(:,2);
					r_star10=(1./betas{pos}(:,2)).*(1+log(1/0.1-1)./betas{pos}(:,1));
					r_star90=(1./betas{pos}(:,2)).*(1+log(1/0.9-1)./betas{pos}(:,1));
					handles1(ell) = plot(deltas{pos},r_star,'Color',clist(ell,:));
					legends1{ell} = ['\sigma=', sprintf('%1.3f', noise_level)];
					hold on

					rel_length = ceil(length(deltas{pos})*9/10);
					ll = ceil(ell*rel_length/length(results_cell)) - 2*ii;
					%ll=ceil(length(deltas{pos})*(1/2)) - 2*ii;
					txt=['\leftarrow n=' sprintf('2^{%d}', log2(n))];
					h=text(deltas{pos}(ll),r_star(ll),txt,'HorizontalAlignment','left', 'FontSize', fs.legend);
					set(h,'Color',clist(ell,:))

					figure(3)
					handles3(ell) = semilogy(deltas{pos},r_star10-r_star90, strcat('-', s_list(ell)), 'Color',clist(ell,:));
					legends3{ell} = ['\sigma=', sprintf('%1.3f', noise_level)];
					txt=['\leftarrow n=' sprintf('2^{%d}', log2(n))];
					% substract 2*ii to position correctly
					h=text(deltas{pos}(ll),r_star10(ll)-r_star90(ll), txt, 'HorizontalAlignment','left', 'FontSize', fs.legend);
					set(h,'Color',clist(ell,:))
					hold on
				else % ens != smv
					fprintf('Error in make_transition_plots: ens & dct code supressed\n');
				end

			end

		end

		figure(1)
		legend(handles1, legends1, 'Location', 'NorthWest', 'FontSize', fs.legend);
		title_txt=[ change_names(alg_list{i}) ' for d = ' num2str(d)];
		title(strrep(title_txt, '_', '\_'), 'FontSize', fs.title)

		xlabel('\delta=m/n','FontSize',fs.axis);
		ylabel('\rho=k/m','FontSize',fs.axis);

		% Tick size
		xt = get(gca, 'XTick');
		set(gca, 'FontSize', fs.ticks);
		xt = get(gca, 'YTick');
		set(gca, 'FontSize', fs.ticks);


		fname1=[destination '/transition_' change_names(alg_list{i}) '_' ens_list{j} '_d_' num2str(d) '_all_n_all_noise'];
		print(strcat(fname1, '.pdf'),'-dpdf')
		print('-depsc', strcat(fname1, '.eps'))
		hold off

		figure(3)
		legend(handles3, legends3, 'FontSize', fs.legend);
		title_txt={'Gap between 10% and 90% success', sprintf('%s, d=%d', change_names(alg_list{i}), d)};
		title(strrep(title_txt, '_', '\_'), 'FontSize', fs.title)
		xlabel('\delta=m/n','FontSize',fs.axis);
		ylabel('\rho=k/m','FontSize',fs.axis);

		% Tick size
		xt = get(gca, 'XTick');
		set(gca, 'FontSize', fs.ticks);
		xt = get(gca, 'YTick');
		set(gca, 'FontSize', fs.ticks);

		fname2=[destination '/transition_width_' change_names(alg_list{i}) '_' ens_list{j} '_d_' num2str(d) '_all_n_all_noise'];
		print(strcat(fname2, '.pdf'),'-dpdf')
		print('-depsc', strcat(fname2, '.eps'))
		hold off

		title_txt
		toc

	end
end


end
