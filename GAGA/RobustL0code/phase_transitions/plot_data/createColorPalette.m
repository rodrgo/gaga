function palette = createColorPalette(alg_list)

	numColors = 9;
	if length(alg_list) > 9
		numColors = length(alg_list);
	end

	palette = colorscale(numColors, 'hue', [1/100 1], 'saturation' , 1, 'value', 0.7);
	palette(1, :) = [0 0 0];
	palette(4, :) = palette(4, :) + [0 -0.2 0];
	idx = 1:numColors;
	idx(1:9) = [1 6 9 8 5 4 2 7 3];
	palette = palette(idx,:);

end
