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