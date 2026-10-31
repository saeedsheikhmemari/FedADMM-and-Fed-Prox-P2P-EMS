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
        env.home(i).evArrivalHour = homes(i).evArrivalHour;
        env.home(i).evDepartureHour = homes(i).evDepartureHour;
        env.home(i).netLoad = homes(i).load(1) - homes(i).pv(1);
        env.home(i).gridExchange = env.home(i).netLoad;
        env.home(i).p2pTrade = 0;
        env.home(i).lastBidPrice = env.localPrice;
        env.home(i).lastBidQty = 0;
    end
end
