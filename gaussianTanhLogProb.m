function logProb = gaussianTanhLogProb(z, mu, logStd, action)
    stdVal = exp(logStd);
    quad = ((z - mu) ./ (stdVal + 1e-8)).^2;
    gaussTerm = -0.5 * (quad + 2 * logStd + log(2 * pi));
    gaussTerm = sum(gaussTerm, 1);
    squashCorr = sum(log(1 - action.^2 + 1e-6), 1);
    logProb = gaussTerm - squashCorr;
end