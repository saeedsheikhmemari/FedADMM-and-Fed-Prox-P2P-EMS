function net = setLearnableValues(net, vals)
    learnables = net.Learnables;
    for k = 1:height(learnables)
        learnables.Value{k} = dlarray(vals{k});
    end
    net.Learnables = learnables;
end