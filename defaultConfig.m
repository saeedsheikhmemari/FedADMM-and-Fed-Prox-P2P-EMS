% =========================================================================
% CONFIGURATION
% =========================================================================
function cfg = defaultConfig()
    cfg.quickDemo = true;

    cfg.numHomes = 10;
    cfg.horizon = 24;
    cfg.stateDim = 9;
    cfg.actionDim = 4;

    % Grid and market pricing
    cfg.gridBuyProfile = [ ...
        0.12 0.12 0.11 0.11 0.11 0.12 0.13 0.15 0.17 0.18 0.19 0.20 ...
        0.20 0.19 0.18 0.19 0.21 0.24 0.27 0.28 0.26 0.22 0.18 0.15];
    cfg.gridSellFrac = 0.70;
    cfg.gridSellProfile = cfg.gridSellFrac * cfg.gridBuyProfile;
    cfg.marketPriceFallbackFrac = 0.50;

    % SAC hyperparameters
    cfg.gamma = 0.99;
    cfg.tau = 0.005;
    cfg.lrActor = 3e-4;
    cfg.lrCritic = 3e-4;
    cfg.lrAlpha = 3e-4;
    cfg.bufferSize = 1e5;
    cfg.batchSize = 128;
    cfg.updateEvery = 1;
    cfg.hiddenUnits = 128;
    cfg.logStdMin = -5;
    cfg.logStdMax = 2;
    cfg.targetEntropy = -4; % 4 continuous actions

    % FedADMM hyperparameters
    cfg.rhoActor = 0.02;
    cfg.rhoCritic = 0.02;
    cfg.k0EpisodesPerSync = 1;

    % Comfort and degradation
    cfg.zetaEV = 1.0;
    cfg.zetaESS = 2.0;
    cfg.deltaESS = 0.01;

    % Device ratings
    cfg.evMaxPower = 7.0;
    cfg.evCap = 50.0;
    cfg.evMinSoC = 10.0;
    cfg.evMaxSoC = 50.0;

    cfg.essMaxPower = 5.0;
    cfg.essCap = 10.0;
    cfg.essMinSoC = 2.0;
    cfg.essMaxSoC = 10.0;

    cfg.bidPriceSpreadFrac = 1.00;

    % Simulate one full day using hour indices 1:24
    cfg.clockHours = 1:24;
    cfg.arrivalHour = 18;
    cfg.departureHour = 7;
    cfg.priceNorm = max(cfg.gridBuyProfile);
    cfg.loadNorm = 6.5;
    cfg.pvNorm = 5.0;

    if cfg.quickDemo
        cfg.numSyncRounds = 4;
        cfg.batchSize = 64;
    else
        cfg.numSyncRounds = 20;
    end
end
