function [km_list best_alg time_ave_list error_ave_list] = make_best_alg_plot(alg_list, ensemble, n, nonzeros, destination, noise_level, SOL_TOL)

tic
if ~strcmp(ensemble, 'smv')
	nonzeros = 0;
end

num_algs = length(alg_list);

if num_algs > 13 
	fprintf('Error: |alg_list| > 13, but only have 13 colors!\n');
end

if true

	% Keep going

	km_cell = cell(num_algs, 1);
	times_cell = cell(num_algs, 1);
	errors_cell = cell(num_algs, 1);
	converged_cell = cell(num_algs, 1);

	for i = 1:num_algs
		%[junk errs junk] = make_delta_rho_plot(alg_list{i},ensemble,n,'l_infinity',0, nonzeros, noise_level);
		[junk errs junk] = make_delta_rho_plot(alg_list{i},ensemble,n,'l_one',0, nonzeros, noise_level);
		[km tt junk] = make_delta_rho_plot(alg_list{i},ensemble,n,'time_for_alg',0, nonzeros, noise_level);
		%[km tt junk] = make_delta_rho_plot(alg_list{i},ensemble,n,'iterations',0, nonzeros);
		errors_cell{i} = errs;
		km_cell{i} = km;
		times_cell{i} = tt;

		% Work out convergence

		ks = km(:, 1);
		ms = km(:, 2);
		if any(ismember({'deterministic_robust_l0', 'robust_l0', 'ssmp_robust'}, {alg_list{i}}))
			mean_err1 = ms*noise_level*sqrt(2/pi);
			sd_err1 = sqrt(ms)*noise_level*sqrt(1 - 2/pi);
			mean_signal_norm = ks*sqrt(2/pi);
			converged = (errs <= (mean_err1 + SOL_TOL*sd_err1)./mean_signal_norm);
		else
			converged = (errs <= SOL_TOL);
		end

		converged_cell{i} = converged;

	end

	km_list = [];
	error_list = [];
	time_list = [];
	converged_list = [];
	for i = 1:num_algs
		km_list = [km_list; km_cell{i}];
		error_list = [error_list; errors_cell{i}];
		time_list = [time_list; times_cell{i}];
		converged_list = [converged_list; converged_cell{i}];
	end

	%make a list of the km_list that have been tested and for which one
	%of the algorithms have an average error less than error_tol

	%error_tol=10^(-2);

	% Note that km_list = [ks, ms];

	%ind=find(error_list<=error_tol);
	ind=find(converged_list == 1);


	km_list=km_list(ind,:);
	km_list=intersect(km_list,km_list,'rows');
	km_list=sortrows(km_list,[2 1]);

	best_alg=cell(size(km_list,1),1);
	time_ave_list=ones(size(km_list,1),1)*max(time_list);
	error_ave_list=zeros(size(km_list,1),1);

	%go through the km_list and see which algorithm has the lowest time 

	for j=1:size(km_list,1)
		for i = 1:num_algs 
			km_list_i = km_cell{i};
			error_list_i = errors_cell{i};
			time_list_i = times_cell{i};
			converged_list_i = converged_cell{i};
			[junk, ind] = intersect(km_list_i, km_list(j, :), 'rows');
			if ~isempty(ind)
				if ((converged_list_i(ind)) & (time_ave_list(j) >= time_list_i(ind)))
				%if ((error_list_i(ind) < error_tol) & (time_ave_list(j) >= time_list_i(ind)))
					time_ave_list(j) = time_list_i(ind);
					error_ave_list(j) = error_list_i(ind);
					best_alg{j} = alg_list{i};
				end
			end
		end
	end

	%-------------------------------------
	% plot a map of which algorithm is best for each kmn

	figure(1)
	hold off

	%c_list = 'rgbcmykw';
	clist = createColorPalette(alg_list);
	s_list = '+o*.xsdh^v><p';
	names_list = {'plus', 'circle', 'asterisk', 'point', 'cross', 'square', 'diamond', 'hexagram', 'up-triangle', 'down-triangle', 'right-triangle', 'left-triangle', 'pentagram'};
	num_best=zeros(13,1);  % MAX SIX ALGOS

	best_ind_list=zeros(length(km_list),1);

	for j=1:length(km_list)
		best_ind=0;
		for q=1:length(alg_list)
			if strcmp(best_alg{j},alg_list(q))
				best_ind=q;
			end
		end
		best_ind_list(j)=best_ind;
		num_best(best_ind)=num_best(best_ind)+1;
		txt=[s_list(best_ind)];
		plot3(km_list(j,2)/n,km_list(j,1)/km_list(j,2),best_ind,txt, 'Color', clist(best_ind,:))
		hold on
	end
	hold off
		
	view([0 90])
	top=0.9;
	for i = 1:num_algs
		if any(best_ind_list == i)
			h = text(0.1, top, 0, strcat(strrep(alg_list{i}, '_', '-'), ': ', names_list{i})); % fix name
			set(h, 'Color', clist(i, :))
			top = top - 0.05;
		end
	end

	axis([0 1 0 1])
	xlabel('\delta=m/n','fontsize',14);
	ylabel('\rho=k/m','fontsize',14);
	 
	title(['Algorithm selection map for d = ' num2str(nonzeros) ' with n = 2^{' num2str(log2(n)) '}' '\sigma=' sprintf('%1.3f', noise_level)],'Fontsize',12)
	fname_out=[destination '/algorithm_selection_' ensemble '_n_' num2str(n) '_nonzeros_' num2str(nonzeros) '_sigma_' sprintf('%1.3f', noise_level)];
	print('-dpdf', strcat(fname_out, '.pdf'))
	print('-depsc', strcat(fname_out, '.eps'))
	 
	%----------------------------------------------
	% make a plot of the time

	figure(2)
	hold off
	set(gca,'fontsize',14)

	a=sort(time_ave_list,'descend');
	c=round(length(a)*0.05);
	thres=a(c);

	ind=find(time_ave_list<thres);
	time_ave_list=time_ave_list(ind);
	km_list=km_list(ind,:);

	delta=km_list(:,2)/n;
	rho=km_list(:,1)./km_list(:,2);

	tri_full=delaunay(delta,rho);

	edge_length=0.15;

	tri=[];

	for j=1:size(tri_full,1)
		d_tri=delta(tri_full(j,:));
		r_tri=rho(tri_full(j,:));
		d_diff=abs([d_tri(2:end)-d_tri(1:end-1); d_tri(1)-d_tri(end)]);
		r_diff=abs([r_tri(2:end)-r_tri(1:end-1); r_tri(1)-r_tri(end)]);
		if max([d_diff])<edge_length
			tri=[tri; tri_full(j,:)];
		end
	end

	tmp=sort(time_ave_list);
	mid=ceil(length(tmp)/2);
	mid=tmp(mid);

	trisurf(tri,delta,rho,min(time_ave_list,2*mid))
	shading interp
	view([0 90])
	axis([0 1 0 1])
	colorbar
	xlabel('\delta=m/n','fontsize',14);
	ylabel('\rho=k/m','fontsize',14);
	 

	title(['Time (ms) of fastest algorithm for d = ' num2str(nonzeros) ' with n = 2^{' num2str(log2(n)) '}' ' \sigma=' sprintf('%1.3f', noise_level)],'Fontsize',12)

	fname_out=[destination '/best_time_' ensemble '_n_' num2str(n) '_nonzeros_' num2str(nonzeros) '_sigma_' sprintf('%1.3f', noise_level)];
	print('-dpdf', strcat(fname_out, '.pdf'))
	print('-depsc', strcat(fname_out, '.eps'))

	%-----------------------------------------
	% compare the time of the "best" with a fixed algorithm

	for i = 1:num_algs
		error_list_i = errors_cell{i};
		km_list_i = km_cell{i};
		time_list_i = times_cell{i};
		converged_list_i = converged_cell{i};

		figure(2 + i)
		hold off
		set(gca,'fontsize',14)

		%ind = find(error_list_i < error_tol);
		ind = find(converged_list_i);
		km_list_i = km_list_i(ind, :);
		%error_list_i = error_list_i(ind);
		converged_list_i = converged_list_i(ind);
		time_list_i = time_list_i(ind);
		[km_list_i, junk, ind] = intersect(km_list, km_list_i, 'rows');
		error_list_i = error_list_i(ind);
		time_list_i = time_list_i(ind);
		converged_list_i = converged_list_i(ind);

		time_ratio_i = zeros(length(km_list_i),1);
		for j = 1:length(time_ratio_i)
			[junk,ind] = intersect(km_list,km_list_i(j,:),'rows');
			time_ratio_i(j) = time_list_i(j)/time_ave_list(ind);
		end

		delta = km_list_i(:,2)/n;
		rho = km_list_i(:,1)./km_list_i(:,2);

		tri_full=delaunay(delta,rho);

		tri=[];

		for j=1:size(tri_full,1)
			d_tri=delta(tri_full(j,:));
			r_tri=rho(tri_full(j,:));
			d_diff=abs([d_tri(2:end)-d_tri(1:end-1); d_tri(1)-d_tri(end)]);
			r_diff=abs([r_tri(2:end)-r_tri(1:end-1); r_tri(1)-r_tri(end)]);
			if max([d_diff])<edge_length 
				tri=[tri; tri_full(j,:)];
			end
		end

		tmp = sort(time_ratio_i);
		mid = ceil(length(tmp)/2);
		mid = tmp(mid);

		trisurf(tri,delta,rho,min(10*mid,time_ratio_i));
		axis([0 1 0 1 0.95 max(time_ratio_i)]);
		shading interp
		view([0 90])
		axis([0 1 0 1])
		colorbar

		xlabel('\delta = m/n','fontsize',14);
		ylabel('\rho = k/m','fontsize',14);

		title(['Time (ms) of ' strrep(alg_list{i}, '_', '-') ' / fastest algorithm for d = ' num2str(nonzeros) ' with n = 2^{' num2str(log2(n)) '}' ' \sigma=' sprintf('%1.3f', noise_level)],'Fontsize',12) % fix name

		fname_out=[destination '/' alg_list{i} '_ratio_' ensemble '_n_' num2str(n) '_nonzeros_' num2str(nonzeros) '_sigma_' sprintf('%1.3f', noise_level)];
		print('-dpdf', strcat(fname_out, '.pdf'))
		print('-depsc', strcat(fname_out, '.eps'))

		ind=find(delta<0.5);
		display(sprintf(strcat(strrep(alg_list{i}, '_', '-'), ' for delta<0.5 never takes more than %f times the time of the fastest algorith'), max(time_ratio_i(ind)))) % fix name
	end

end
