function state = getStateForHome(env, i, cfg)
    hour = env.hours(env.t);
    timeToDep = hoursUntilDeparture(hour, env.home(i).evDepartureHour);

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