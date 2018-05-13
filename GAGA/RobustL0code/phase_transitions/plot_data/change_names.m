function new_name = change_name(old_name)
	if strcmp(old_name, 'robust_l0')
		new_name = 'rl0-Q';
	elseif strcmp(old_name, 'robust_l0_trans')
		new_name = 'rl0';
	elseif strcmp(old_name, 'robust_l0_adaptive_trans')
		new_name = 'rl0-A';
	elseif strcmp(old_name, 'robust_l0_adaptive')
		new_name = 'rl0-AQ';
	elseif strcmp(old_name, 'ssmp_robust')
		new_name = 'ssmp';
	elseif strcmp(old_name, 'smp_robust')
		new_name = 'smp';
	elseif strcmp(old_name, 'cgiht_robust')
		new_name = 'cgiht';
	else
		new_name = old_name;
	end
end
