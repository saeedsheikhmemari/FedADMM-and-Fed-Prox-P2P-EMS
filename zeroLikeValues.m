function zerosVals = zeroLikeValues(vals)
    zerosVals = cell(size(vals));
    for k = 1:numel(vals)
        zerosVals{k} = zeros(size(vals{k}), 'like', vals{k});
    end
end