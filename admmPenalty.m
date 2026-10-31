function p = admmPenalty(net, globalValues, dualValues, rho)
    p = 0;
    for k = 1:height(net.Learnables)
        localVal = net.Learnables.Value{k};
        globalVal = globalValues{k};
        dualVal = dualValues{k};

        diffVal = localVal - globalVal;
        p = p + sum(dualVal .* diffVal, 'all') + 0.5 * rho * sum(diffVal.^2, 'all');
    end
end
