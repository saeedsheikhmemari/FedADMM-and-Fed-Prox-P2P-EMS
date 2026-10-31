function states = getAllStates(env, cfg)
    states = zeros(cfg.stateDim, cfg.numHomes);
    for i = 1:cfg.numHomes
        states(:, i) = getStateForHome(env, i, cfg);
    end
end