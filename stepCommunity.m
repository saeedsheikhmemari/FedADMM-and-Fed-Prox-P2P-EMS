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
        evAvailable = isEVAvailable(hour, env.home(i).evArrivalHour, env.home(i).evDepartureHour);

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

        tradeFrac = 0.5 * (aBidQty + 1);   % maps [-1,1] -> [0,1]

        if preNet > 0
            % deficit home -> buyer
            desiredQty = tradeFrac * preNet;
            bidPrice(i) = mapBuyerBidPrice(aBidPrice, gridSell, gridBuy);
        elseif preNet < 0
            % surplus home -> seller
            desiredQty = -tradeFrac * abs(preNet);
            bidPrice(i) = mapSellerAskPrice(aBidPrice, gridSell, gridBuy);
        else
            desiredQty = 0;
            bidPrice(i) = 0.5 * (gridBuy + gridSell);
        end

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
    market.numBuyers = nnz(desiredBidQty > 1e-9);
    market.numSellers = nnz(desiredBidQty < -1e-9);
    market.totalBuyRequest = sum(max(desiredBidQty, 0));
    market.totalSellRequest = sum(max(-desiredBidQty, 0));

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
        if hour == env.home(i).evDepartureHour
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
    info.numBuyers = market.numBuyers;
    info.numSellers = market.numSellers;
    info.totalBuyRequest = market.totalBuyRequest;
    info.totalSellRequest = market.totalSellRequest;
    info.market = market;
    info.home = homeInfo;

    env.t = min(env.t + 1, cfg.horizon);
    nextStates = getAllStates(env, cfg);
end