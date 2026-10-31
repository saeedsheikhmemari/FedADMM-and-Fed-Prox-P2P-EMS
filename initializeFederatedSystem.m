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
