clc
clear


% RUN_FEDADMM_SAC_TENHOME
% FedADMM-SAC skeleton for 10 heterogeneous homes with PV, BESS, EV, and P2P trading.
%
% Key change requested:
%   - k0 is implemented as LOCAL EPISODES PER ADMM SYNCHRONIZATION.
%   - Number of homes is increased from 2 to 10.
%
% State per home:
%   [SoC_EV; SoC_BESS; PV; FixedLoad; GridBuyPrice; GridSellPrice; PrevLocalP2PPrice; HourNorm; TimeToDepartureNorm]
%
% Action per home:
%   [aEV; aBESS; aBidQty; aBidPrice], all in [-1, 1]
%
% This is a research-oriented MATLAB implementation draft. It keeps:
%   1) 24-hour day-ahead environment
%   2) SAC actor/critic with replay buffer and target critic
%   3) FedADMM-style consensus penalty with dual variables
%   4) ADMM synchronization every k0 local episodes
%
% Notes:
%   - Target critic and entropy temperature remain local.
%   - All 10 homes participate in every synchronization round.
%   - The code is written conservatively for MATLAB compatibility by editing
%     Learnables tables directly rather than relying on dlupdate(net,...).

    rng(11, 'twister');

    cfg = defaultConfig();
    homes = buildTenHomeSystem(cfg);
    [globalModel, agents] = initializeFederatedSystem(cfg);

    history.syncMeanReward = zeros(cfg.numSyncRounds, 1);
    history.syncMeanImport = zeros(cfg.numSyncRounds, 1);
    history.syncMeanPrice = zeros(cfg.numSyncRounds, 1);
    history.episodeReward = zeros(cfg.numSyncRounds * cfg.k0EpisodesPerSync, cfg.numHomes);

    episodeCounter = 0;
    fprintf('Starting FedADMM-SAC training for %d homes.\n', cfg.numHomes);
    fprintf('k0 = %d local episodes per ADMM synchronization.\n', cfg.k0EpisodesPerSync);

    for syncRound = 1:cfg.numSyncRounds
        fprintf('ADMM sync round %d / %d\n', syncRound, cfg.numSyncRounds);

        % Broadcast current global model to all homes at the start of the sync round
        for i = 1:cfg.numHomes
            agents(i).actor = setLearnableValues(agents(i).actor, globalModel.actorValues);
            agents(i).critic = setLearnableValues(agents(i).critic, globalModel.criticValues);
            agents(i).targetCritic = setLearnableValues(agents(i).targetCritic, globalModel.criticValues);
        end

        localEpisodeRewards = zeros(cfg.k0EpisodesPerSync, cfg.numHomes);
        localMeanPrice = zeros(cfg.k0EpisodesPerSync, 1);
        localMeanImport = zeros(cfg.k0EpisodesPerSync, 1);

        % ---------------------------
        % k0 local episodes per sync
        % ---------------------------
        for e = 1:cfg.k0EpisodesPerSync
            env = resetEnvironment(homes, cfg);
            episodeReward = zeros(cfg.numHomes, 1);
            priceSeries = zeros(cfg.horizon, 1);
            importSeries = zeros(cfg.horizon, cfg.numHomes);

            for t = 1:cfg.horizon
                states = getAllStates(env, cfg);
                actions = zeros(cfg.actionDim, cfg.numHomes);

                for i = 1:cfg.numHomes
                    [a, ~] = sampleActorAction(agents(i).actor, states(:, i), cfg, false);
                    actions(:, i) = a;
                end

                [env, nextStates, rewards, dones, info] = stepCommunity(env, actions, cfg);

                for i = 1:cfg.numHomes
                    agents(i).buffer = pushTransition(agents(i).buffer, ...
                        states(:, i), actions(:, i), rewards(i), nextStates(:, i), dones(i));
                    episodeReward(i) = episodeReward(i) + rewards(i);
                    importSeries(t, i) = info.home(i).gridImport;
                end

                priceSeries(t) = info.localPrice;

                for i = 1:cfg.numHomes
                    if agents(i).buffer.size >= cfg.batchSize && mod(agents(i).totalEnvSteps, cfg.updateEvery) == 0
                        [agents(i), ~] = localSACUpdateADMM(agents(i), globalModel, cfg);
                    end
                    agents(i).totalEnvSteps = agents(i).totalEnvSteps + 1;
                end

                if all(dones)
                    break;
                end
            end

            episodeCounter = episodeCounter + 1;
            history.episodeReward(episodeCounter, :) = episodeReward.';
            localEpisodeRewards(e, :) = episodeReward.';
            localMeanPrice(e) = mean(priceSeries);
            localMeanImport(e) = mean(importSeries(:));
        end

        % ---------------------------
        % ADMM synchronization step
        % ---------------------------
        globalModel.actorValues = admmGlobalUpdate(agents, 'actor', 'actorDual', cfg.rhoActor);
        globalModel.criticValues = admmGlobalUpdate(agents, 'critic', 'criticDual', cfg.rhoCritic);

        globalModel.actor = setLearnableValues(globalModel.actor, globalModel.actorValues);
        globalModel.critic = setLearnableValues(globalModel.critic, globalModel.criticValues);

        for i = 1:cfg.numHomes
            agents(i).actorDual = dualAscentUpdate(agents(i).actorDual, agents(i).actor, globalModel.actorValues, cfg.rhoActor);
            agents(i).criticDual = dualAscentUpdate(agents(i).criticDual, agents(i).critic, globalModel.criticValues, cfg.rhoCritic);
        end

        history.syncMeanReward(syncRound) = mean(localEpisodeRewards, 'all');
        history.syncMeanImport(syncRound) = mean(localMeanImport);
        history.syncMeanPrice(syncRound) = mean(localMeanPrice);

        fprintf('  mean reward: %.4f | mean grid import: %.4f kW | mean local P2P price: %.4f $/kWh\n', ...
            history.syncMeanReward(syncRound), history.syncMeanImport(syncRound), history.syncMeanPrice(syncRound));
    end

    fprintf('Training complete. Running deterministic evaluation...\n');
    evalResults = evaluateDeterministicPolicy(globalModel.actor, homes, cfg);
    baselineResults = evaluateBaselineNoHEMS(homes, cfg);
    plotResults(history, evalResults, baselineResults, cfg);

    results.cfg = cfg;
    results.homes = homes;
    results.history = history;
    results.globalModel = globalModel;
    results.evaluation = evalResults;
    results.baseline = baselineResults;

    fprintf('Done.\n');


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
    cfg.k0EpisodesPerSync = 2;

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

% =========================================================================
% TEN-HOME DATA
% =========================================================================
function homes = buildTenHomeSystem(cfg)
    baseLoad1 = [ ...
        1.25 1.20 1.15 1.10 1.10 1.20 1.35 1.60 1.80 1.95 2.05 2.10 ...
        2.00 1.95 2.05 2.20 2.50 3.10 3.80 3.60 4.05 3.70 2.80 1.90];

    baseLoad2 = [ ...
        0.95 0.90 0.85 0.82 0.84 0.92 1.05 1.20 1.35 1.45 1.55 1.60 ...
        1.52 1.48 1.55 1.68 1.95 2.25 2.60 2.45 2.80 2.55 1.95 1.30];

    basePV1 = 10*[ ...
        0.00 0.00 0.00 0.00 0.00 0.10 0.30 0.80 1.60 2.70 3.80 4.60 ...
        5.00 4.70 4.00 3.00 1.80 0.80 0.20 0.00 0.00 0.00 0.00 0.00];

    basePV2 = 10*[ ...
        0.00 0.00 0.00 0.00 0.00 0.00 0.20 0.60 1.30 2.30 3.20 4.00 ...
        4.60 4.30 3.60 2.60 1.50 0.50 0.00 0.00 0.00 0.00 0.00 0.00];

    eveningShape = zeros(1, 24);
    eveningShape([18 19 20 21 22]) = [0.08 0.15 0.25 0.18 0.10];

    for i = 1:cfg.numHomes
        if mod(i, 2) == 1
            loadBase = baseLoad1;
            pvBase = basePV1;
        else
            loadBase = baseLoad2;
            pvBase = basePV2;
        end

        loadScale = 0.88 + 0.045 * (i - 1);
        pvScale = 0.82 + 0.03 * (i - 1);
        pvShift = 0.03 * sin(((1:24) + i) / 24 * 2 * pi);
        loadBias = 0.03 * cos(((1:24) + 2 * i) / 24 * 2 * pi);

        loadProfile = loadScale * loadBase + 0.10 * (i - 1) / (cfg.numHomes - 1) * eveningShape + loadBias;
        loadProfile = max(loadProfile, 0.5);

        pvProfile = pvScale * pvBase + pvShift;
        pvProfile = min(max(pvProfile, 0.0), cfg.pvNorm);

        homes(i).name = sprintf('H%02d', i);
        homes(i).load = loadProfile;
        homes(i).pv = pvProfile;
        homes(i).evEta = 0.89 + 0.005 * mod(i - 1, 3);
        homes(i).essEta = 0.94 + 0.003 * mod(i - 1, 4);
        homes(i).initialEVSoC = min(18 + 2.5 * (i - 1), 42);
        homes(i).initialESSSoC = min(3.5 + 0.45 * mod(i - 1, 8), cfg.essMaxSoC - 0.5);
    end
end


% =========================================================================
% ENVIRONMENT
% =========================================================================
function env = resetEnvironment(homes, cfg)
    env.t = 1;
    env.done = false;
    env.hours = cfg.clockHours;
    env.localPrice = 0.5 * (cfg.gridBuyProfile(1) + cfg.gridSellProfile(1));
    env.totalDemand = 0;
    env.totalGridImport = 0;
    env.totalGridExport = 0;
    env.totalP2PVolume = 0;

    for i = 1:cfg.numHomes
        env.home(i).name = homes(i).name;
        env.home(i).load = homes(i).load;
        env.home(i).pv = homes(i).pv;
        env.home(i).evEta = homes(i).evEta;
        env.home(i).essEta = homes(i).essEta;
        env.home(i).socEV = homes(i).initialEVSoC;
        env.home(i).socESS = homes(i).initialESSSoC;
        env.home(i).netLoad = homes(i).load(1) - homes(i).pv(1);
        env.home(i).gridExchange = env.home(i).netLoad;
        env.home(i).p2pTrade = 0;
        env.home(i).lastBidPrice = env.localPrice;
        env.home(i).lastBidQty = 0;
    end
end

function states = getAllStates(env, cfg)
    states = zeros(cfg.stateDim, cfg.numHomes);
    for i = 1:cfg.numHomes
        states(:, i) = getStateForHome(env, i, cfg);
    end
end

function state = getStateForHome(env, i, cfg)
    hour = env.hours(env.t);
    timeToDep = hoursUntilDeparture(hour, cfg.departureHour);

    state = [
        env.home(i).socEV / cfg.evCap
        env.home(i).socESS / cfg.essCap
        env.home(i).pv(env.t) / cfg.pvNorm
        env.home(i).load(env.t) / cfg.loadNorm
        cfg.gridBuyProfile(env.t) / cfg.priceNorm
        cfg.gridSellProfile(env.t) / cfg.priceNorm
        env.localPrice / cfg.priceNorm
        (hour - 1) / 23
        timeToDep / 24
    ];
end

function [env, nextStates, rewards, dones, info] = stepCommunity(env, actions, cfg)
    t = env.t;
    hour = env.hours(t);
    gridBuy = cfg.gridBuyProfile(t);
    gridSell = cfg.gridSellProfile(t);

    homeInfo = repmat(struct(), cfg.numHomes, 1);
    preMarketNet = zeros(cfg.numHomes, 1);
    desiredBidQty = zeros(cfg.numHomes, 1);
    bidPrice = zeros(cfg.numHomes, 1);

    for i = 1:cfg.numHomes
        evAvailable = isEVAvailable(hour, cfg.arrivalHour, cfg.departureHour);

        aEV = boundAction(actions(1, i));
        aESS = boundAction(actions(2, i));
        aBidQty = boundAction(actions(3, i));
        aBidPrice = boundAction(actions(4, i));

        if ~evAvailable
            aEV = 0;
        end

        pEV = aEV * cfg.evMaxPower;
        pESS = aESS * cfg.essMaxPower;

        prevSoCEV = env.home(i).socEV;
        prevSoCESS = env.home(i).socESS;

        socEV = prevSoCEV + pEV * env.home(i).evEta;
        socESS = prevSoCESS + pESS * env.home(i).essEta;

        preNet = env.home(i).load(t) + pEV + pESS - env.home(i).pv(t);
        preMarketNet(i) = preNet;

        if preNet > 0
            desiredQty = max(0, aBidQty) * preNet;
        elseif preNet < 0
            desiredQty = -max(0, -aBidQty) * abs(preNet);
        else
            desiredQty = 0;
        end

        bidPrice(i) = mapBidPrice(aBidPrice, gridSell, gridBuy, cfg);
        desiredBidQty(i) = desiredQty;

        homeInfo(i).aEV = aEV;
        homeInfo(i).aESS = aESS;
        homeInfo(i).aBidQty = aBidQty;
        homeInfo(i).aBidPrice = aBidPrice;
        homeInfo(i).pEV = pEV;
        homeInfo(i).pESS = pESS;
        homeInfo(i).prevSoCEV = prevSoCEV;
        homeInfo(i).prevSoCESS = prevSoCESS;
        homeInfo(i).socEV = socEV;
        homeInfo(i).socESS = socESS;
        homeInfo(i).preMarketNet = preNet;
        homeInfo(i).desiredBidQty = desiredQty;
        homeInfo(i).bidPrice = bidPrice(i);
    end

    market = clearDoubleAuction(desiredBidQty, bidPrice, gridBuy, gridSell, cfg);

    totalDemand = 0;
    totalGridImport = 0;
    totalGridExport = 0;
    rewards = zeros(cfg.numHomes, 1);
    dones = false(cfg.numHomes, 1);

    for i = 1:cfg.numHomes
        qP2P = market.qP2P(i);
        gridExchange = preMarketNet(i) - qP2P;
        gridImport = max(gridExchange, 0);
        gridExport = max(-gridExchange, 0);

        socEV = homeInfo(i).socEV;
        socESS = homeInfo(i).socESS;

        evPenalty = 0;
        if hour == cfg.departureHour
            evPenalty = cfg.zetaEV * max(cfg.evMaxSoC - socEV, 0);
        end

        if socESS > cfg.essMaxSoC
            essPenalty = cfg.zetaESS * (socESS - cfg.essMaxSoC)^2 + cfg.deltaESS * abs(homeInfo(i).pESS);
        elseif socESS < cfg.essMinSoC
            essPenalty = cfg.zetaESS * (cfg.essMinSoC - socESS)^2 + cfg.deltaESS * abs(homeInfo(i).pESS);
        else
            essPenalty = cfg.deltaESS * abs(homeInfo(i).pESS);
        end

        p2pCost = market.localPrice * max(qP2P, 0) - market.localPrice * max(-qP2P, 0);
        gridCost = gridBuy * gridImport - gridSell * gridExport;
        totalCost = p2pCost + gridCost + evPenalty + essPenalty;
        rewards(i) = -totalCost;

        env.home(i).socEV = min(max(socEV, cfg.evMinSoC), cfg.evMaxSoC);
        env.home(i).socESS = min(max(socESS, cfg.essMinSoC), cfg.essMaxSoC);
        env.home(i).netLoad = preMarketNet(i);
        env.home(i).gridExchange = gridExchange;
        env.home(i).p2pTrade = qP2P;
        env.home(i).lastBidPrice = bidPrice(i);
        env.home(i).lastBidQty = desiredBidQty(i);

        homeInfo(i).p2pTrade = qP2P;
        homeInfo(i).gridExchange = gridExchange;
        homeInfo(i).gridImport = gridImport;
        homeInfo(i).gridExport = gridExport;
        homeInfo(i).netLoad = preMarketNet(i);
        homeInfo(i).reward = rewards(i);

        totalDemand = totalDemand + preMarketNet(i);
        totalGridImport = totalGridImport + gridImport;
        totalGridExport = totalGridExport + gridExport;
        dones(i) = (t >= cfg.horizon);
    end

    env.localPrice = market.localPrice;
    env.totalDemand = totalDemand;
    env.totalGridImport = totalGridImport;
    env.totalGridExport = totalGridExport;
    env.totalP2PVolume = market.totalVolume;

    info.localPrice = market.localPrice;
    info.gridBuyPrice = gridBuy;
    info.gridSellPrice = gridSell;
    info.totalDemand = totalDemand;
    info.totalGridImport = totalGridImport;
    info.totalGridExport = totalGridExport;
    info.totalP2PVolume = market.totalVolume;
    info.market = market;
    info.home = homeInfo;

    env.t = min(env.t + 1, cfg.horizon);
    nextStates = getAllStates(env, cfg);
end

function market = clearDoubleAuction(desiredBidQty, bidPrice, gridBuy, gridSell, cfg)
    numHomes = numel(desiredBidQty);
    qP2P = zeros(numHomes, 1);

    buyerIdx = find(desiredBidQty > 1e-9);
    sellerIdx = find(desiredBidQty < -1e-9);

    buyQty = desiredBidQty(buyerIdx);
    sellQty = -desiredBidQty(sellerIdx);
    buyPrice = bidPrice(buyerIdx);
    sellPrice = bidPrice(sellerIdx);

    [buyPriceSorted, buyOrder] = sort(buyPrice, 'descend');
    [sellPriceSorted, sellOrder] = sort(sellPrice, 'ascend');

    buyerIdx = buyerIdx(buyOrder);
    sellerIdx = sellerIdx(sellOrder);
    buyQty = buyQty(buyOrder);
    sellQty = sellQty(sellOrder);

    bi = 1;
    si = 1;
    tradedValue = 0;
    totalVolume = 0;

    while bi <= numel(buyerIdx) && si <= numel(sellerIdx)
        if buyPriceSorted(bi) + 1e-9 < sellPriceSorted(si)
            break;
        end

        tradedQty = min(buyQty(bi), sellQty(si));
        if tradedQty <= 0
            break;
        end

        clearingPrice = 0.5 * (buyPriceSorted(bi) + sellPriceSorted(si));

        qP2P(buyerIdx(bi)) = qP2P(buyerIdx(bi)) + tradedQty;
        qP2P(sellerIdx(si)) = qP2P(sellerIdx(si)) - tradedQty;

        tradedValue = tradedValue + clearingPrice * tradedQty;
        totalVolume = totalVolume + tradedQty;

        buyQty(bi) = buyQty(bi) - tradedQty;
        sellQty(si) = sellQty(si) - tradedQty;

        if buyQty(bi) <= 1e-9
            bi = bi + 1;
        end
        if sellQty(si) <= 1e-9
            si = si + 1;
        end
    end

    if totalVolume > 0
        localPrice = tradedValue / totalVolume;
    else
        localPrice = cfg.marketPriceFallbackFrac * (gridBuy + gridSell);
    end

    market.localPrice = localPrice;
    market.qP2P = qP2P;
    market.totalVolume = totalVolume;
end

function bidPrice = mapBidPrice(actionValue, gridSell, gridBuy, cfg)
    lo = gridSell;
    hi = gridBuy;
    frac = 0.5 * (actionValue + 1);
    bidPrice = lo + frac * cfg.bidPriceSpreadFrac * (hi - lo);
    bidPrice = min(max(bidPrice, lo), hi);
end

function tf = isEVAvailable(hour, arrivalHour, departureHour)
    tf = (hour >= arrivalHour) || (hour < departureHour);
end

function dt = hoursUntilDeparture(currentHour, departureHour)
    if currentHour <= departureHour
        dt = departureHour - currentHour;
    else
        dt = (24 - currentHour) + departureHour;
    end
end

function a = boundAction(a)
    a = max(-1, min(1, a));
end

% =========================================================================
% AGENTS / GLOBAL MODEL
% =========================================================================
function [globalModel, agents] = initializeFederatedSystem(cfg)
    globalModel.actor = buildActorNetwork(cfg.stateDim, cfg.actionDim, cfg.hiddenUnits);
    globalModel.critic = buildCriticNetwork(cfg.stateDim + cfg.actionDim, cfg.hiddenUnits);
    globalModel.actorValues = getLearnableValues(globalModel.actor);
    globalModel.criticValues = getLearnableValues(globalModel.critic);

    for i = 1:cfg.numHomes
        agents(i).actor = setLearnableValues(buildActorNetwork(cfg.stateDim, cfg.actionDim, cfg.hiddenUnits), globalModel.actorValues);
        agents(i).critic = setLearnableValues(buildCriticNetwork(cfg.stateDim + cfg.actionDim, cfg.hiddenUnits), globalModel.criticValues);
        agents(i).targetCritic = setLearnableValues(buildCriticNetwork(cfg.stateDim + cfg.actionDim, cfg.hiddenUnits), globalModel.criticValues);

        agents(i).buffer = createReplayBuffer(cfg.bufferSize, cfg.stateDim, cfg.actionDim);

        agents(i).actorAvgGrad = [];
        agents(i).actorAvgSqGrad = [];
        agents(i).criticAvgGrad = [];
        agents(i).criticAvgSqGrad = [];
        agents(i).alphaAvgGrad = 0;
        agents(i).alphaAvgSqGrad = 0;

        agents(i).logAlpha = log(0.2);
        agents(i).updateCount = 0;
        agents(i).totalEnvSteps = 0;

        agents(i).actorDual = zeroLikeValues(globalModel.actorValues);
        agents(i).criticDual = zeroLikeValues(globalModel.criticValues);
    end
end

function actor = buildActorNetwork(stateDim, actionDim, hiddenUnits)
    layers = [
        featureInputLayer(stateDim, Normalization='none', Name='state')
        fullyConnectedLayer(hiddenUnits, Name='fc1')
        reluLayer(Name='relu1')
        fullyConnectedLayer(hiddenUnits, Name='fc2')
        reluLayer(Name='relu2')
        fullyConnectedLayer(2 * actionDim, Name='policyOut')
    ];
    actor = dlnetwork(layerGraph(layers));
end

function critic = buildCriticNetwork(inputDim, hiddenUnits)
    layers = [
        featureInputLayer(inputDim, Normalization='none', Name='sa')
        fullyConnectedLayer(hiddenUnits, Name='fc1')
        reluLayer(Name='relu1')
        fullyConnectedLayer(hiddenUnits, Name='fc2')
        reluLayer(Name='relu2')
        fullyConnectedLayer(1, Name='qOut')
    ];
    critic = dlnetwork(layerGraph(layers));
end

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

function logProb = gaussianTanhLogProb(z, mu, logStd, action)
    stdVal = exp(logStd);
    quad = ((z - mu) ./ (stdVal + 1e-8)).^2;
    gaussTerm = -0.5 * (quad + 2 * logStd + log(2 * pi));
    gaussTerm = sum(gaussTerm, 1);
    squashCorr = sum(log(1 - action.^2 + 1e-6), 1);
    logProb = gaussTerm - squashCorr;
end

function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end

% =========================================================================
% REPLAY BUFFER
% =========================================================================
function buffer = createReplayBuffer(capacity, stateDim, actionDim)
    buffer.state = zeros(stateDim, capacity);
    buffer.action = zeros(actionDim, capacity);
    buffer.reward = zeros(1, capacity);
    buffer.nextState = zeros(stateDim, capacity);
    buffer.done = zeros(1, capacity);

    buffer.capacity = capacity;
    buffer.ptr = 1;
    buffer.size = 0;
end

function buffer = pushTransition(buffer, state, action, reward, nextState, doneFlag)
    idx = buffer.ptr;

    buffer.state(:, idx) = state;
    buffer.action(:, idx) = action;
    buffer.reward(1, idx) = reward;
    buffer.nextState(:, idx) = nextState;
    buffer.done(1, idx) = double(doneFlag);

    buffer.ptr = idx + 1;
    if buffer.ptr > buffer.capacity
        buffer.ptr = 1;
    end
    buffer.size = min(buffer.size + 1, buffer.capacity);
end

function batch = sampleMiniBatch(buffer, batchSize)
    idx = randperm(buffer.size, batchSize);
    batch.state = buffer.state(:, idx);
    batch.action = buffer.action(:, idx);
    batch.reward = buffer.reward(:, idx);
    batch.nextState = buffer.nextState(:, idx);
    batch.done = buffer.done(:, idx);
end

% =========================================================================
% LOCAL SAC UPDATE WITH ADMM PENALTY
% =========================================================================
function [agent, metrics] = localSACUpdateADMM(agent, globalModel, cfg)
    batch = sampleMiniBatch(agent.buffer, cfg.batchSize);
    alphaVal = exp(agent.logAlpha);

    dlState = dlarray(batch.state, 'CB');
    dlAction = dlarray(batch.action, 'CB');
    dlReward = dlarray(batch.reward, 'CB');
    dlNextState = dlarray(batch.nextState, 'CB');
    dlDone = dlarray(batch.done, 'CB');

    [criticLoss, criticGradients] = dlfeval(@criticLossFnADMM, ...
        agent.critic, agent.targetCritic, agent.actor, ...
        dlState, dlAction, dlReward, dlNextState, dlDone, ...
        alphaVal, globalModel.criticValues, agent.criticDual, cfg);

    agent.updateCount = agent.updateCount + 1;
    [agent.critic, agent.criticAvgGrad, agent.criticAvgSqGrad] = adamupdate( ...
        agent.critic, criticGradients, ...
        agent.criticAvgGrad, agent.criticAvgSqGrad, ...
        agent.updateCount, cfg.lrCritic);

    [actorLoss, actorGradients, meanLogPi] = dlfeval(@actorLossFnADMM, ...
        agent.actor, agent.critic, dlState, ...
        alphaVal, globalModel.actorValues, agent.actorDual, cfg);

    [agent.actor, agent.actorAvgGrad, agent.actorAvgSqGrad] = adamupdate( ...
        agent.actor, actorGradients, ...
        agent.actorAvgGrad, agent.actorAvgSqGrad, ...
        agent.updateCount, cfg.lrActor);

    alphaGrad = -(meanLogPi + cfg.targetEntropy);
    [agent.logAlpha, agent.alphaAvgGrad, agent.alphaAvgSqGrad] = adamupdate( ...
        agent.logAlpha, alphaGrad, ...
        agent.alphaAvgGrad, agent.alphaAvgSqGrad, ...
        agent.updateCount, cfg.lrAlpha);

    agent.targetCritic = softUpdateTarget(agent.targetCritic, agent.critic, cfg.tau);

    metrics.criticLoss = double(extractdata(criticLoss));
    metrics.actorLoss = double(extractdata(actorLoss));
    metrics.alpha = exp(agent.logAlpha);
end

function [loss, gradients] = criticLossFnADMM(critic, targetCritic, actor, ...
    dlState, dlAction, dlReward, dlNextState, dlDone, ...
    alphaVal, globalCriticValues, criticDualValues, cfg)

    [nextAction, nextLogPi] = sampleActorActionBatch(actor, dlNextState, cfg);
    qNext = forward(targetCritic, [dlNextState; nextAction]);
    y = dlReward + cfg.gamma * (1 - dlDone) .* (qNext - alphaVal * nextLogPi);

    qPred = forward(critic, [dlState; dlAction]);
    mseLoss = mean((qPred - y).^2, 'all');
    admmLoss = admmPenalty(critic, globalCriticValues, criticDualValues, cfg.rhoCritic);

    loss = mseLoss + admmLoss;
    gradients = dlgradient(loss, critic.Learnables);
end

function [loss, gradients, meanLogPi] = actorLossFnADMM(actor, critic, dlState, ...
    alphaVal, globalActorValues, actorDualValues, cfg)

    [piAction, logPi] = sampleActorActionBatch(actor, dlState, cfg);
    qPi = forward(critic, [dlState; piAction]);

    baseLoss = mean(alphaVal * logPi - qPi, 'all');
    admmLoss = admmPenalty(actor, globalActorValues, actorDualValues, cfg.rhoActor);

    loss = baseLoss + admmLoss;
    gradients = dlgradient(loss, actor.Learnables);

    meanLogPi = double(gather(extractdata(mean(logPi, 'all'))));
end

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

function dualValues = dualAscentUpdate(dualValues, localNet, globalValues, rho)
    for k = 1:numel(dualValues)
        localVal = extractdata(localNet.Learnables.Value{k});
        dualValues{k} = dualValues{k} + rho * (localVal - globalValues{k});
    end
end

function targetCritic = softUpdateTarget(targetCritic, critic, tau)
    targetLearnables = targetCritic.Learnables;
    sourceLearnables = critic.Learnables;

    for k = 1:height(targetLearnables)
        targetLearnables.Value{k} = (1 - tau) * targetLearnables.Value{k} + tau * sourceLearnables.Value{k};
    end

    targetCritic.Learnables = targetLearnables;
end

function vals = getLearnableValues(net)
    vals = cell(height(net.Learnables), 1);
    for k = 1:height(net.Learnables)
        vals{k} = extractdata(net.Learnables.Value{k});
    end
end

function net = setLearnableValues(net, vals)
    learnables = net.Learnables;
    for k = 1:height(learnables)
        learnables.Value{k} = dlarray(vals{k});
    end
    net.Learnables = learnables;
end

function zerosVals = zeroLikeValues(vals)
    zerosVals = cell(size(vals));
    for k = 1:numel(vals)
        zerosVals{k} = zeros(size(vals{k}), 'like', vals{k});
    end
end


% =========================================================================
% EVALUATION
% =========================================================================
function evalResults = evaluateDeterministicPolicy(actor, homes, cfg)
    env = resetEnvironment(homes, cfg);

    evalResults.hours = cfg.clockHours;
    evalResults.localPrice = zeros(cfg.horizon, 1);
    evalResults.gridBuyPrice = zeros(cfg.horizon, 1);
    evalResults.gridSellPrice = zeros(cfg.horizon, 1);
    evalResults.totalDemand = zeros(cfg.horizon, 1);
    evalResults.totalGridImport = zeros(cfg.horizon, 1);
    evalResults.totalGridExport = zeros(cfg.horizon, 1);
    evalResults.totalP2PVolume = zeros(cfg.horizon, 1);

    evalResults.netLoad = zeros(cfg.horizon, cfg.numHomes);
    evalResults.gridExchange = zeros(cfg.horizon, cfg.numHomes);
    evalResults.gridImport = zeros(cfg.horizon, cfg.numHomes);
    evalResults.gridExport = zeros(cfg.horizon, cfg.numHomes);
    evalResults.p2pTrade = zeros(cfg.horizon, cfg.numHomes);
    evalResults.bidPrice = zeros(cfg.horizon, cfg.numHomes);
    evalResults.bidQty = zeros(cfg.horizon, cfg.numHomes);

    evalResults.actionEV = zeros(cfg.horizon, cfg.numHomes);
    evalResults.actionESS = zeros(cfg.horizon, cfg.numHomes);
    evalResults.pEV = zeros(cfg.horizon, cfg.numHomes);
    evalResults.pESS = zeros(cfg.horizon, cfg.numHomes);
    evalResults.load = zeros(cfg.horizon, cfg.numHomes);
    evalResults.pv = zeros(cfg.horizon, cfg.numHomes);
    evalResults.socEV = zeros(cfg.horizon, cfg.numHomes);
    evalResults.socESS = zeros(cfg.horizon, cfg.numHomes);
    evalResults.reward = zeros(cfg.horizon, cfg.numHomes);

    for t = 1:cfg.horizon
        states = getAllStates(env, cfg);
        actions = zeros(cfg.actionDim, cfg.numHomes);

        for i = 1:cfg.numHomes
            [a, ~] = sampleActorAction(actor, states(:, i), cfg, true);
            actions(:, i) = a;
        end

        [env, ~, rewards, ~, info] = stepCommunity(env, actions, cfg);

        evalResults.localPrice(t) = info.localPrice;
        evalResults.gridBuyPrice(t) = info.gridBuyPrice;
        evalResults.gridSellPrice(t) = info.gridSellPrice;
        evalResults.totalDemand(t) = info.totalDemand;
        evalResults.totalGridImport(t) = info.totalGridImport;
        evalResults.totalGridExport(t) = info.totalGridExport;
        evalResults.totalP2PVolume(t) = info.totalP2PVolume;
        evalResults.reward(t, :) = rewards.';

        for i = 1:cfg.numHomes
            evalResults.netLoad(t, i) = info.home(i).netLoad;
            evalResults.gridExchange(t, i) = info.home(i).gridExchange;
            evalResults.gridImport(t, i) = info.home(i).gridImport;
            evalResults.gridExport(t, i) = info.home(i).gridExport;
            evalResults.p2pTrade(t, i) = info.home(i).p2pTrade;
            evalResults.bidPrice(t, i) = info.home(i).bidPrice;
            evalResults.bidQty(t, i) = info.home(i).desiredBidQty;

            evalResults.actionEV(t, i) = info.home(i).aEV;
            evalResults.actionESS(t, i) = info.home(i).aESS;
            evalResults.pEV(t, i) = info.home(i).pEV;
            evalResults.pESS(t, i) = info.home(i).pESS;
            evalResults.load(t, i) = homes(i).load(t);
            evalResults.pv(t, i) = homes(i).pv(t);
            evalResults.socEV(t, i) = env.home(i).socEV;
            evalResults.socESS(t, i) = env.home(i).socESS;
        end
    end
end

function baselineResults = evaluateBaselineNoHEMS(homes, cfg)
    baselineResults.hours = cfg.clockHours;
    baselineResults.localPrice = zeros(cfg.horizon, 1);
    baselineResults.gridBuyPrice = cfg.gridBuyProfile(:);
    baselineResults.gridSellPrice = cfg.gridSellProfile(:);
    baselineResults.totalDemand = zeros(cfg.horizon, 1);
    baselineResults.totalGridImport = zeros(cfg.horizon, 1);
    baselineResults.totalGridExport = zeros(cfg.horizon, 1);
    baselineResults.totalP2PVolume = zeros(cfg.horizon, 1);

    baselineResults.netLoad = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.gridExchange = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.gridImport = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.gridExport = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.p2pTrade = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.bidPrice = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.bidQty = zeros(cfg.horizon, cfg.numHomes);

    baselineResults.pEV = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.pESS = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.load = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.pv = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.socEV = zeros(cfg.horizon, cfg.numHomes);
    baselineResults.socESS = zeros(cfg.horizon, cfg.numHomes);

    socEV = zeros(cfg.numHomes, 1);
    for i = 1:cfg.numHomes
        socEV(i) = homes(i).initialEVSoC;
    end

    for t = 1:cfg.horizon
        hour = cfg.clockHours(t);
        totalDemand = 0;
        totalGridImport = 0;
        totalGridExport = 0;

        for i = 1:cfg.numHomes
            baselineResults.load(t, i) = homes(i).load(t);
            baselineResults.pv(t, i) = homes(i).pv(t);
            baselineResults.pESS(t, i) = 0.0;
            baselineResults.socESS(t, i) = homes(i).initialESSSoC;

            if isEVAvailable(hour, cfg.arrivalHour, cfg.departureHour)
                pEV = max(0.0, min(cfg.evMaxPower, (cfg.evMaxSoC - socEV(i)) / homes(i).evEta));
            else
                pEV = 0.0;
            end

            socEV(i) = min(cfg.evMaxSoC, socEV(i) + pEV * homes(i).evEta);
            baselineResults.pEV(t, i) = pEV;
            baselineResults.socEV(t, i) = socEV(i);
            baselineResults.netLoad(t, i) = homes(i).load(t) + pEV - homes(i).pv(t);
            baselineResults.gridExchange(t, i) = baselineResults.netLoad(t, i);
            baselineResults.gridImport(t, i) = max(baselineResults.gridExchange(t, i), 0);
            baselineResults.gridExport(t, i) = max(-baselineResults.gridExchange(t, i), 0);

            totalDemand = totalDemand + baselineResults.netLoad(t, i);
            totalGridImport = totalGridImport + baselineResults.gridImport(t, i);
            totalGridExport = totalGridExport + baselineResults.gridExport(t, i);
        end

        baselineResults.totalDemand(t) = totalDemand;
        baselineResults.totalGridImport(t) = totalGridImport;
        baselineResults.totalGridExport(t) = totalGridExport;
    end
end

function plotResults(history, evalResults, baselineResults, cfg)
    figure('Name', 'FedADMM-SAC P2P Summary', 'Color', 'w');

    subplot(2, 3, 1);
    plot(1:cfg.numSyncRounds, history.syncMeanReward, '-o', 'LineWidth', 1.25);
    xlabel('ADMM synchronization round');
    ylabel('Mean episode reward');
    title(sprintf('Training summary (k0 = %d episodes/sync)', cfg.k0EpisodesPerSync));
    grid on;

    subplot(2, 3, 2);
    plot(evalResults.hours, evalResults.localPrice, '-o', 'LineWidth', 1.25); hold on;
    plot(evalResults.hours, evalResults.gridBuyPrice, '--', 'LineWidth', 1.10);
    plot(evalResults.hours, evalResults.gridSellPrice, ':', 'LineWidth', 1.10);
    xlabel('Hour (1-24)');
    ylabel('Price ($/kWh)');
    title('Local market price and grid tariffs');
    legend({'Local P2P price', 'Grid buy tariff', 'Grid sell tariff'}, 'Location', 'best');
    grid on;

    subplot(2, 3, 3);
    bar(evalResults.hours, evalResults.totalP2PVolume, 0.75);
    xlabel('Hour (1-24)');
    ylabel('Matched P2P energy (kWh)');
    title('Total community P2P traded volume');
    grid on;

    subplot(2, 3, 4);
    plot(evalResults.hours, evalResults.totalGridImport, '-s', 'LineWidth', 1.25); hold on;
    plot(evalResults.hours, -evalResults.totalGridExport, '-d', 'LineWidth', 1.25);
    plot(baselineResults.hours, baselineResults.totalGridImport, '--', 'LineWidth', 1.10);
    plot(baselineResults.hours, -baselineResults.totalGridExport, '--', 'LineWidth', 1.10);
    yline(0, ':');
    xlabel('Hour (1-24)');
    ylabel('Community grid exchange (kW)');
    title('Grid import/export comparison');
    legend({'Import (P2P FedADMM-SAC)', 'Export (P2P FedADMM-SAC)', 'Import (Baseline)', 'Export (Baseline)'}, 'Location', 'best');
    grid on;

    subplot(2, 3, 5);
    plot(evalResults.hours, evalResults.totalDemand, '-^', 'LineWidth', 1.25); hold on;
    plot(baselineResults.hours, baselineResults.totalDemand, '--', 'LineWidth', 1.10);
    xlabel('Hour (1-24)');
    ylabel('Total pre-market net load (kW)');
    title('Aggregated net demand before trading');
    legend({'P2P FedADMM-SAC', 'Baseline'}, 'Location', 'best');
    grid on;

    subplot(2, 3, 6);
    plot(evalResults.hours, mean(evalResults.socEV, 2), '-o', 'LineWidth', 1.25); hold on;
    plot(evalResults.hours, mean(evalResults.socESS, 2), '-x', 'LineWidth', 1.25);
    plot(baselineResults.hours, mean(baselineResults.socEV, 2), '--', 'LineWidth', 1.10);
    xlabel('Hour (1-24)');
    ylabel('Average SoC (kWh)');
    title('Average EV and BESS SoC across 10 homes');
    legend({'Mean EV SoC (P2P FedADMM-SAC)', 'Mean BESS SoC (P2P FedADMM-SAC)', 'Mean EV SoC (Baseline)'}, 'Location', 'best');
    grid on;

    plotFigure6StyleP2P(evalResults, baselineResults, cfg);
end

function plotFigure6StyleP2P(evalResults, baselineResults, cfg)
    figure('Name', 'Figure-6-Style P2P Market Comparison', 'Color', 'w');
    tl = tiledlayout(5, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'Per-home market behavior: P2P FedADMM-SAC (left) vs no-HEMS baseline (right)');

    hours = evalResults.hours(:);

    for i = 1:cfg.numHomes
        nexttile;
        plotHomeMarketPanel(hours, evalResults.netLoad(:, i), evalResults.p2pTrade(:, i), ...
            evalResults.gridExchange(:, i), sprintf('H%02d - P2P FedADMM-SAC', i), i == 1);

        nexttile;
        plotHomeMarketPanel(hours, baselineResults.netLoad(:, i), baselineResults.p2pTrade(:, i), ...
            baselineResults.gridExchange(:, i), sprintf('H%02d - Baseline', i), false);
    end
end

function plotHomeMarketPanel(hours, netLoadSeries, p2pTradeSeries, gridExchangeSeries, panelTitle, showLegend)
    plot(hours, netLoadSeries, '-', 'LineWidth', 1.15); hold on;
    plot(hours, p2pTradeSeries, '--', 'LineWidth', 1.15);
    plot(hours, gridExchangeSeries, '-', 'LineWidth', 1.35);
    yline(0, ':');
    xlim([1 24]);
    xticks([1 6 12 18 24]);
    xlabel('Hour');
    ylabel('kW');
    title(panelTitle);
    grid on;
    if showLegend
        legend({'Pre-market net load', 'P2P traded energy', 'Residual grid exchange'}, 'Location', 'best');
    end
end
