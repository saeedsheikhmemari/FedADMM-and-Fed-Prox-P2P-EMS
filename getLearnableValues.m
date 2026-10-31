function vals = getLearnableValues(net)
    vals = cell(height(net.Learnables), 1);
    for k = 1:height(net.Learnables)
        vals{k} = extractdata(net.Learnables.Value{k});
    end
end