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
        evalResults.numBuyers(t) = info.numBuyers;
        evalResults.numSellers(t) = info.numSellers;
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