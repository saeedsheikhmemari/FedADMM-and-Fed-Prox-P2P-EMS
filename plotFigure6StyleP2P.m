
function plotFigure6StyleP2P(evalResults, baselineResults, cfg)
    figure('Name', 'Figure-6-Style P2P Market Comparison', 'Color', 'w');
    tl = tiledlayout(5, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'Per-home market behavior: P2P FedADMM-SAC (left) vs no-HEMS baseline (right)');

    hours = evalResults.hours(:);

    for i = 1:cfg.numHomes
        nexttile;
        plotHomeMarketPanel(hours, evalResults.netLoad(:, i), evalResults.p2pTrade(:, i), ...
            evalResults.gridExchange(:, i), sprintf('H%02d - P2P FedADMM-SAC', i), i == 1);

        nexttile;
        plotHomeMarketPanel(hours, baselineResults.netLoad(:, i), baselineResults.p2pTrade(:, i), ...
            baselineResults.gridExchange(:, i), sprintf('H%02d - Baseline', i), false);
    end
end

