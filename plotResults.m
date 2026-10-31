
function plotResults(history, evalResults, baselineResults, cfg)
    figure('Name', 'FedADMM-SAC P2P Summary', 'Color', 'w');

    subplot(2, 3, 1);
    plot(1:cfg.numSyncRounds, history.syncMeanReward, '-o', 'LineWidth', 1.25);
    xlabel('ADMM synchronization round');
    ylabel('Mean episode reward');
    title(sprintf('Training summary (k0 = %d episodes/sync)', cfg.k0EpisodesPerSync));
    grid on;

    subplot(2, 3, 2);
    plot(evalResults.hours, evalResults.localPrice, '-o', 'LineWidth', 1.25); hold on;
    plot(evalResults.hours, evalResults.gridBuyPrice, '--', 'LineWidth', 1.10);
    plot(evalResults.hours, evalResults.gridSellPrice, ':', 'LineWidth', 1.10);
    xlabel('Hour (1-24)');
    ylabel('Price ($/kWh)');
    title('Local market price and grid tariffs');
    legend({'Local P2P price', 'Grid buy tariff', 'Grid sell tariff'}, 'Location', 'best');
    grid on;

    subplot(2, 3, 3);
    bar(evalResults.hours, evalResults.totalP2PVolume, 0.75);
    xlabel('Hour (1-24)');
    ylabel('Matched P2P energy (kWh)');
    title('Total community P2P traded volume');
    grid on;

    subplot(2, 3, 4);
    plot(evalResults.hours, evalResults.totalGridImport, '-s', 'LineWidth', 1.25); hold on;
    plot(evalResults.hours, -evalResults.totalGridExport, '-d', 'LineWidth', 1.25);
    plot(baselineResults.hours, baselineResults.totalGridImport, '--', 'LineWidth', 1.10);
    plot(baselineResults.hours, -baselineResults.totalGridExport, '--', 'LineWidth', 1.10);
    yline(0, ':');
    xlabel('Hour (1-24)');
    ylabel('Community grid exchange (kW)');
    title('Grid import/export comparison');
    legend({'Import (P2P FedADMM-SAC)', 'Export (P2P FedADMM-SAC)', 'Import (Baseline)', 'Export (Baseline)'}, 'Location', 'best');
    grid on;

    subplot(2, 3, 5);
    plot(evalResults.hours, evalResults.totalDemand, '-^', 'LineWidth', 1.25); hold on;
    plot(baselineResults.hours, baselineResults.totalDemand, '--', 'LineWidth', 1.10);
    xlabel('Hour (1-24)');
    ylabel('Total pre-market net load (kW)');
    title('Aggregated net demand before trading');
    legend({'P2P FedADMM-SAC', 'Baseline'}, 'Location', 'best');
    grid on;

    subplot(2, 3, 6);
    plot(evalResults.hours, mean(evalResults.socEV, 2), '-o', 'LineWidth', 1.25); hold on;
    plot(evalResults.hours, mean(evalResults.socESS, 2), '-x', 'LineWidth', 1.25);
    plot(baselineResults.hours, mean(baselineResults.socEV, 2), '--', 'LineWidth', 1.10);
    xlabel('Hour (1-24)');
    ylabel('Average SoC (kWh)');
    title('Average EV and BESS SoC across 10 homes');
    legend({'Mean EV SoC (P2P FedADMM-SAC)', 'Mean BESS SoC (P2P FedADMM-SAC)', 'Mean EV SoC (Baseline)'}, 'Location', 'best');
    grid on;

    plotFigure6StyleP2P(evalResults, baselineResults, cfg);
end
