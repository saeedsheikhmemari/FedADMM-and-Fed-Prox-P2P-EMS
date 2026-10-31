% =========================================================================
% ACTION SAMPLING
% =========================================================================
function [action, aux] = sampleActorAction(actor, state, cfg, deterministic)
    dlS = dlarray(state, 'CB');
    raw = predict(actor, dlS);

    mu = raw(1:cfg.actionDim, :);
    logStd = raw(cfg.actionDim+1:end, :);
    logStd = clamp(logStd, cfg.logStdMin, cfg.logStdMax);
    stdVal = exp(logStd);

    if deterministic
        z = mu;
    else
        epsVal = randn(size(extractdata(mu)));
        z = mu + stdVal .* epsVal;
    end

    a = tanh(z);
    logProb = gaussianTanhLogProb(z, mu, logStd, a);

    action = extractdata(a(:, 1));
    aux.mu = extractdata(mu(:, 1));
    aux.logStd = extractdata(logStd(:, 1));
    aux.logProb = extractdata(logProb(1));
end
