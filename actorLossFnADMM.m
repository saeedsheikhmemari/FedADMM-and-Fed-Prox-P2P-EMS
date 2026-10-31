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