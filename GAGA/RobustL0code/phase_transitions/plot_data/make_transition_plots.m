function make_transition_plots(alg_list, ens_list, destination)

name_split = strsplit(destination, '/');
tag = name_split{end};

tic
format shortg

%clist='brkgmcybrk';
clist = createColorPalette(alg_list);

% ================
% One plot per noise_level, one curve per n
% ================

for j=1:length(ens_list)
	for i=1:length(alg_list)
		fname=['results_' alg_list{i} '_S_' ens_list{j}];
		load(fname)

		% Spit by noise_level

		for ell = 1:length(results_cell)

			% Extract relevant data

			betas = results_cell{ell}.betas;
			deltas = results_cell{ell}.deltas;
			n_list = results_cell{ell}.n_list;
			nz_list = results_cell{ell}.nz_list;
			noise_level = results_cell{ell}.noise_level;

			% Keep going

			figure(1)
			set(gca,'fontsize',14)
			hold off

			figure(3)
			set(gca,'fontsize',14)
			hold off

			if strcmp(ens_list{j},'smv')
				nz_list_full=nz_list;
				nz_list=intersect(nz_list,nz_list);
				for k=1:length(nz_list)
					ind=find(nz_list_full==nz_list(k));
					n=n_list(ind);
					for zz=1:length(ind)
						figure(1)

						r_star=1./betas{ind(zz)}(:,2);
						r_star10=(1./betas{ind(zz)}(:,2)).*(1+log(1/0.1-1)./betas{ind(zz)}(:,1));
						r_star90=(1./betas{ind(zz)}(:,2)).*(1+log(1/0.9-1)./betas{ind(zz)}(:,1));
						plot(deltas{ind(zz)},r_star,'Color',clist(zz,:))
						hold on

						ll=ceil(length(deltas{ind(zz)})*(zz-1/2)/length(ind));
						txt=['\leftarrow n=2^{' num2str(log2(n(zz))) '}'];
						h=text(deltas{ind(zz)}(ll),r_star(ll),txt,'HorizontalAlignment','left');
						set(h,'Color',clist(zz,:))

						figure(3)
						plot(deltas{ind(zz)},r_star10-r_star90,'Color',clist(zz,:));
						txt=['\leftarrow n=2^{' num2str(log2(n(zz))) '}'];
						h=text(deltas{ind(zz)}(ll),r_star10(ll)-r_star90(ll),txt,'HorizontalAlignment','left');
						set(h,'Color',clist(zz,:))
						hold on
					end
					figure(1)
					title_txt=[ alg_list{i} ' for d = ' num2str(nz_list(k)) ' \sigma = ' num2str(noise_level)];
					title(strrep(title_txt, '_', '\_'))

					xlabel('\delta=m/n','fontsize',14);
					ylabel('\rho=k/m','fontsize',14);
					fname1=[destination '/transition_' alg_list{i} '_' ens_list{j} '_d_' num2str(nz_list(k)) '_noise_' num2str(noise_level)];
					print(strcat(fname1, '.pdf'),'-dpdf')
					print('-depsc', strcat(fname1, '.eps'))
					hold off

					figure(3)
					title_txt=['Gap between 10% and 90% success for ' alg_list{i} ' d = ' num2str(nz_list(k)) ' \sigma = ' num2str(noise_level)];
					title(strrep(title_txt, '_', '\_'))
					xlabel('\delta=m/n','fontsize',14);
					ylabel('\rho=k/m','fontsize',14);
					fname2=[destination '/transition_width_' alg_list{i} '_' ens_list{j} '_d_' num2str(nz_list(k)) '_noise_' num2str(noise_level)];
					print(strcat(fname2, '.pdf'),'-dpdf')
					print('-depsc', strcat(fname2, '.eps'))
					hold off
				end
			else % ens != smv
				fprintf('Error in make_transition_plots: ens & dct code supressed\n');
			end

			title_txt
			toc

		end
	end
end

% ================
% One plot per n, several curves per noise_level
% ================

for j=1:length(ens_list)
	for i=1:length(alg_list)
		fname=['results_' alg_list{i} '_S_' ens_list{j}];
		load(fname)

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

			% Create plots

			figure(1)
			set(gca,'fontsize',14)
			hold off

			figure(3)
			set(gca,'fontsize',14)
			hold off

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
					plot(deltas{pos},r_star,'Color',clist(ell,:))
					hold on

					ll=ceil(length(deltas{pos})*(zz-1/2)/length(ind));
					txt=['\leftarrow \sigma=' sprintf('%1.3f', noise_level)];
					h=text(deltas{pos}(ll),r_star(ll),txt,'HorizontalAlignment','left');
					set(h,'Color',clist(ell,:))

					figure(3)
					plot(deltas{pos},r_star10-r_star90,'Color',clist(ell,:));
					txt=['\leftarrow \sigma=' sprintf('%1.3f', noise_level)];
					h=text(deltas{pos}(ll),r_star10(ll)-r_star90(ll),txt,'HorizontalAlignment','left');
					set(h,'Color',clist(ell,:))
					hold on
				else % ens != smv
					fprintf('Error in make_transition_plots: ens & dct code supressed\n');
				end

			end

			figure(1)
			title_txt=[ alg_list{i} ' for d = ' num2str(d) ' and n = ' num2str(n)];
			title(strrep(title_txt, '_', '\_'))

			xlabel('\delta=m/n','fontsize',14);
			ylabel('\rho=k/m','fontsize',14);
			fname1=[destination '/transition_' alg_list{i} '_' ens_list{j} '_d_' num2str(d) '_n_' num2str(n)];
			print(strcat(fname1, '.pdf'),'-dpdf')
			print('-depsc', strcat(fname1, '.eps'))
			hold off

			figure(3)
			title_txt=['Gap between 10% and 90% success for ' alg_list{i} ' d = ' num2str(d)];
			title(strrep(title_txt, '_', '\_'))
			xlabel('\delta=m/n','fontsize',14);
			ylabel('\rho=k/m','fontsize',14);
			fname2=[destination '/transition_width_' alg_list{i} '_' ens_list{j} '_d_' num2str(d) '_n_' num2str(n)];
			print(strcat(fname2, '.pdf'),'-dpdf')
			print('-depsc', strcat(fname2, '.eps'))
			hold off

			title_txt
			toc

		end

	end
end

% ================
% One plot per algorithm, for all n, for all noise_levels
% ================

% Create plots

figure(1)
set(gca,'fontsize',14)
hold off

handles1 = zeros(length(results_cell), 1);
legends1 = cell(length(results_cell), 1);

figure(3)
set(gca,'fontsize',14)
hold off

handles3 = zeros(length(results_cell), 1);
legends3 = cell(length(results_cell), 1);

for j=1:length(ens_list)
	for i=1:length(alg_list)
		fname=['results_' alg_list{i} '_S_' ens_list{j}];
		load(fname)

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

					ll=ceil(length(deltas{pos})*(zz-1/2)/length(ind)) - 2*ii;
					txt=['\leftarrow n=' sprintf('2^{%d}', log2(n))];
					h=text(deltas{pos}(ll),r_star(ll),txt,'HorizontalAlignment','left');
					set(h,'Color',clist(ell,:))

					figure(3)
					handles3(ell) = semilogy(deltas{pos},r_star10-r_star90,'Color',clist(ell,:));
					legends3{ell} = ['\sigma=', sprintf('%1.3f', noise_level)];
					txt=['\leftarrow n=' sprintf('2^{%d}', log2(n))];
					% substract 2*ii to position correctly
					h=text(deltas{pos}(ll - 2*ell),r_star10(ll - 2*ell)-r_star90(ll - 2*ell),txt,'HorizontalAlignment','left');
					set(h,'Color',clist(ell,:))
					hold on
				else % ens != smv
					fprintf('Error in make_transition_plots: ens & dct code supressed\n');
				end

			end

		end

		figure(1)
		legend(handles1, legends1, 'Location', 'NorthWest');
		title_txt=[ alg_list{i} ' for d = ' num2str(d)];
		title(strrep(title_txt, '_', '\_'))

		xlabel('\delta=m/n','fontsize',14);
		ylabel('\rho=k/m','fontsize',14);
		fname1=[destination '/transition_' alg_list{i} '_' ens_list{j} '_d_' num2str(d) '_all_n_all_noise'];
		print(strcat(fname1, '.pdf'),'-dpdf')
		print('-depsc', strcat(fname1, '.eps'))
		hold off

		figure(3)
		legend(handles3, legends3);
		title_txt=['Gap between 10% and 90% success for ' alg_list{i} ' d = ' num2str(d)];
		title(strrep(title_txt, '_', '\_'))
		xlabel('\delta=m/n','fontsize',14);
		ylabel('\rho=k/m','fontsize',14);
		fname2=[destination '/transition_width_' alg_list{i} '_' ens_list{j} '_d_' num2str(d) '_all_n_all_noise'];
		print(strcat(fname2, '.pdf'),'-dpdf')
		print('-depsc', strcat(fname2, '.eps'))
		hold off

		title_txt
		toc

	end
end

% end function
end
