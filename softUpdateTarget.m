function targetCritic = softUpdateTarget(targetCritic, critic, tau)
    targetLearnables = targetCritic.Learnables;
    sourceLearnables = critic.Learnables;

    for k = 1:height(targetLearnables)
        targetLearnables.Value{k} = (1 - tau) * targetLearnables.Value{k} + tau * sourceLearnables.Value{k};
    end

    targetCritic.Learnables = targetLearnables;
end
