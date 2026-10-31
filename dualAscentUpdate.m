function dualValues = dualAscentUpdate(dualValues, localNet, globalValues, rho)
    for k = 1:numel(dualValues)
        localVal = extractdata(localNet.Learnables.Value{k});
        dualValues{k} = dualValues{k} + rho * (localVal - globalValues{k});
    end
end