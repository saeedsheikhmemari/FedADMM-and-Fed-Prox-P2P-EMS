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

            if isEVAvailable(hour, homes(i).evArrivalHour, homes(i).evDepartureHour)
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