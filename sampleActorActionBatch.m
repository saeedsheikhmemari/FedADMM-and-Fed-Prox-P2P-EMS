function [action, logProb] = sampleActorActionBatch(actor, dlStateBatch, cfg)
    raw = forward(actor, dlStateBatch);

    mu = raw(1:cfg.actionDim, :);
    logStd = raw(cfg.actionDim+1:end, :);
    logStd = clamp(logStd, cfg.logStdMin, cfg.logStdMax);
    stdVal = exp(logStd);

    epsVal = randn(size(extractdata(mu)));
    z = mu + stdVal .* epsVal;
    action = tanh(z);
    logProb = gaussianTanhLogProb(z, mu, logStd, action);
end
