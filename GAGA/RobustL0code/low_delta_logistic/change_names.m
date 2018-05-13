function new_name = change_name(old_name)
	if strcmp(old_name, 'robust-l0')
		new_name = 'rl0-Q';
	elseif strcmp(old_name, 'robust-l0-trans')
		new_name = 'rl0';
	elseif strcmp(old_name, 'robust-l0-adaptive-trans')
		new_name = 'rl0-A';
	elseif strcmp(old_name, 'robust-l0-adaptive')
		new_name = 'rl0-AQ';
	elseif strcmp(old_name, 'ssmp-robust')
		new_name = 'ssmp';
	elseif strcmp(old_name, 'smp-robust')
		new_name = 'smp';
	elseif strcmp(old_name, 'cgiht-robust')
		new_name = 'cgiht';
	else
		new_name = old_name;
	end
end
