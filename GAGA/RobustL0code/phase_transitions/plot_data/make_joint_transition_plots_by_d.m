function make_joint_transition_plots_by_d(alg, ens, n, nonzeros_list, destination)
% ens should be 'dct', 'smv', or 'gen'
% for ens=='smv' the third (nonzeros) argument is required

name_split = strsplit(destination, '/');
tag = name_split{end};

if (strcmp(ens,'smv')==1 & nargin<3)
  warning('smv requires nonzeros to be specified')
end

%clist='brkmgcybrk';
clist = colorscale(length(nonzeros_list), 'hue', [1/(length(nonzeros_list) + 1) 1], 'saturation' , 0.9, 'value', 0.9);
colorList = colorscale(9, 'hue', [1/100 1], 'saturation' , 1, 'value', 0.7);
colorList(1, :) = [0 0 0];
colorList(4, :) = colorList(4, :) + [0 -0.2 0];
colorList = colorList([1 6 9 8 5 4 2 7],:);
clist = colorList;

figure(1)
hold off

if true
  fname=['results_' alg '_S_' ens];
  load(fname)

	for j = 1:length(nonzeros_list)
		nonzeros = nonzeros_list(j);
	
		ind_n=find(n_list==n);
		if strcmp(ens,'smv')
			ind_nz=find(nz_list==nonzeros);
			ind_n=intersect(ind_n,ind_nz);
		end
		
		plot(deltas{ind_n},1./betas{ind_n}(:,2),'Color',clist(j,:));
		ll=ceil(length(deltas{ind_n})*(j-1/2)/length(nonzeros_list));
		txt=['\leftarrow ' sprintf('d = %d', nonzeros)];
		h=text(deltas{ind_n}(ll),1./betas{ind_n}(ll,2),txt,'HorizontalAlignment','left');
		set(h,'color',clist(j,:))
		hold on

	end
end

fname1=[destination '/transition_' alg '_' ens];
if strcmp(ens,'smv')
	fname1=[fname1 '_several_d'];
end

if strcmp(ens,'smv')
	txt=['50% phase transition curves for ' ...
		alg ' with n = 2^{' num2str(log2(n)) '}'];
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
