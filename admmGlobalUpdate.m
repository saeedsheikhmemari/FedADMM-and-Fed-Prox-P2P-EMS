% =========================================================================
% ADMM UTILITIES
% =========================================================================
function globalValues = admmGlobalUpdate(agents, netField, dualField, rho)
    templateNet = agents(1).(netField);
    numRows = height(templateNet.Learnables);
    globalValues = cell(numRows, 1);

    for row = 1:numRows
        acc = 0;
        for i = 1:numel(agents)
            localVal = extractdata(agents(i).(netField).Learnables.Value{row});
            dualVal = agents(i).(dualField){row};
            acc = acc + rho * localVal + dualVal;
        end
        globalValues{row} = acc ./ (numel(agents) * rho);
    end
end