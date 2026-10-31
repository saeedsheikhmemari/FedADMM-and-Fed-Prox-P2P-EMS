
clc
clear
% RUN_FEDADMM_SAC_TENHOME_P2P_HETERO
% FedADMM-SAC skeleton for 10 heterogeneous homes with PV, BESS, EV, and P2P trading.
%
% Key change requested:
%   - k0 is implemented as LOCAL EPISODES PER ADMM SYNCHRONIZATION.
%   - Number of homes is increased from 2 to 10.
%
% State per home:
%   [SoC_EV; SoC_BESS; PV; FixedLoad; GridBuyPrice; GridSellPrice; PrevLocalP2PPrice; HourNorm; TimeToDepartureNorm]
%
% Action per home:
%   [aEV; aBESS; aBidQty; aBidPrice], all in [-1, 1]
%
% This is a research-oriented MATLAB implementation draft. It keeps:
%   1) 24-hour day-ahead environment
%   2) SAC actor/critic with replay buffer and target critic
%   3) FedADMM-style consensus penalty with dual variables
%   4) ADMM synchronization every k0 local episodes
%
% Notes
%   - Target critic and entropy temperature remain local.
%   - All 10 homes participate in every synchronization round.
%   - The code is written conservatively for MATLAB compatibility by editing
%     Learnables tables directly rather than relying on dlupdate(net,...).

rng(11, 'twister');

cfg = defaultConfig();
homes = buildTenHomeSystem(cfg);
[globalModel, agents] = initializeFederatedSystem(cfg);

history.syncMeanReward = zeros(cfg.numSyncRounds, 1);
history.syncMeanImport = zeros(cfg.numSyncRounds, 1);
history.syncMeanPrice = zeros(cfg.numSyncRounds, 1);
history.episodeReward = zeros(cfg.numSyncRounds * cfg.k0EpisodesPerSync, cfg.numHomes);

episodeCounter = 0;
fprintf('Starting FedADMM-SAC training for %d homes.\n', cfg.numHomes);
fprintf('k0 = %d local episodes per ADMM synchronization.\n', cfg.k0EpisodesPerSync);

for syncRound = 1:cfg.numSyncRounds
    fprintf('ADMM sync round %d / %d\n', syncRound, cfg.numSyncRounds);

    % Broadcast current global model to all homes at the start of the sync round
    for i = 1:cfg.numHomes
        agents(i).actor = setLearnableValues(agents(i).actor, globalModel.actorValues);
        agents(i).critic = setLearnableValues(agents(i).critic, globalModel.criticValues);
        agents(i).targetCritic = setLearnableValues(agents(i).targetCritic, globalModel.criticValues);
    end

    localEpisodeRewards = zeros(cfg.k0EpisodesPerSync, cfg.numHomes);
    localMeanPrice = zeros(cfg.k0EpisodesPerSync, 1);
    localMeanImport = zeros(cfg.k0EpisodesPerSync, 1);

    % ---------------------------
    % k0 local episodes per sync
    % ---------------------------
    for e = 1:cfg.k0EpisodesPerSync
        env = resetEnvironment(homes, cfg);
        episodeReward = zeros(cfg.numHomes, 1);
        priceSeries = zeros(cfg.horizon, 1);
        importSeries = zeros(cfg.horizon, cfg.numHomes);

        for t = 1:cfg.horizon
            states = getAllStates(env, cfg);
            actions = zeros(cfg.actionDim, cfg.numHomes);

            for i = 1:cfg.numHomes
                [a, ~] = sampleActorAction(agents(i).actor, states(:, i), cfg, false);
                actions(:, i) = a;
            end

            [env, nextStates, rewards, dones, info] = stepCommunity(env, actions, cfg);

            for i = 1:cfg.numHomes
                agents(i).buffer = pushTransition(agents(i).buffer, ...
                    states(:, i), actions(:, i), rewards(i), nextStates(:, i), dones(i));
                episodeReward(i) = episodeReward(i) + rewards(i);
                importSeries(t, i) = info.home(i).gridImport;
            end

            priceSeries(t) = info.localPrice;

            for i = 1:cfg.numHomes
                if agents(i).buffer.size >= cfg.batchSize && mod(agents(i).totalEnvSteps, cfg.updateEvery) == 0
                    [agents(i), ~] = localSACUpdateADMM(agents(i), globalModel, cfg);
                end
                agents(i).totalEnvSteps = agents(i).totalEnvSteps + 1;
            end

            if all(dones)
                break;
            end
        end

        episodeCounter = episodeCounter + 1;
        history.episodeReward(episodeCounter, :) = episodeReward.';
        localEpisodeRewards(e, :) = episodeReward.';
        localMeanPrice(e) = mean(priceSeries);
        localMeanImport(e) = mean(importSeries(:));
    end

    % ---------------------------
    % ADMM synchronization step
    % ---------------------------
    globalModel.actorValues = admmGlobalUpdate(agents, 'actor', 'actorDual', cfg.rhoActor);
    globalModel.criticValues = admmGlobalUpdate(agents, 'critic', 'criticDual', cfg.rhoCritic);

    globalModel.actor = setLearnableValues(globalModel.actor, globalModel.actorValues);
    globalModel.critic = setLearnableValues(globalModel.critic, globalModel.criticValues);

    for i = 1:cfg.numHomes
        agents(i).actorDual = dualAscentUpdate(agents(i).actorDual, agents(i).actor, globalModel.actorValues, cfg.rhoActor);
        agents(i).criticDual = dualAscentUpdate(agents(i).criticDual, agents(i).critic, globalModel.criticValues, cfg.rhoCritic);
    end

    history.syncMeanReward(syncRound) = mean(localEpisodeRewards, 'all');
    history.syncMeanImport(syncRound) = mean(localMeanImport);
    history.syncMeanPrice(syncRound) = mean(localMeanPrice);

    fprintf('  mean reward: %.4f | mean grid import: %.4f kW | mean local P2P price: %.4f $/kWh\n', ...
        history.syncMeanReward(syncRound), history.syncMeanImport(syncRound), history.syncMeanPrice(syncRound));
end

fprintf('Training complete. Running deterministic evaluation...\n');
evalResults = evaluateDeterministicPolicy(globalModel.actor, homes, cfg);
baselineResults = evaluateBaselineNoHEMS(homes, cfg);
plotResults(history, evalResults, baselineResults, cfg);

results.cfg = cfg;
results.homes = homes;
results.history = history;
results.globalModel = globalModel;
results.evaluation = evalResults;
results.baseline = baselineResults;

fprintf('Done.\n');

% End


