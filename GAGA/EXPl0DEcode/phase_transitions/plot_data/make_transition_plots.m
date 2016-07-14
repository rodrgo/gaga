function make_transition_plots(alg_list, ens_list, destination)

name_split = strsplit(destination, '/');
tag = name_split{end};

tic
format shortg

%clist='brkgmcybrk';
clist = createColorPalette(alg_list);

for j=1:length(ens_list)
	for i=1:length(alg_list)
		fname=['results_' alg_list{i} '_S_' ens_list{j}];
		load(fname)

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
				title_txt=[ alg_list{i} ' for d = ' num2str(nz_list(k)) ' '];
				title(strrep(title_txt, '_', '\_'))

				xlabel('\delta=m/n','fontsize',14);
				ylabel('\rho=k/m','fontsize',14);
				fname1=[destination '/transition_' alg_list{i} '_' ens_list{j} '_d_' num2str(nz_list(k))];
				print(fname1,'-dpdf')
				print('-depsc', strcat(fname1, '.eps'))
				hold off

				figure(3)
				title_txt=['Gap between 10% and 90% success for ' alg_list{i} ' d = ' num2str(nz_list(k)) ' '];
				title(strrep(title_txt, '_', '\_'))
				xlabel('\delta=m/n','fontsize',14);
				ylabel('\rho=k/m','fontsize',14);
				fname2=[destination '/transition_width_' alg_list{i} '_' ens_list{j} '_d_' num2str(nz_list(k))];
				print(fname2,'-dpdf')
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

% end function
end
