function plotHomeMarketPanel(hours, netLoadSeries, p2pTradeSeries, gridExchangeSeries, panelTitle, showLegend)
    plot(hours, netLoadSeries, '-', 'LineWidth', 1.15); hold on;
    plot(hours, p2pTradeSeries, '--', 'LineWidth', 1.15);
    plot(hours, gridExchangeSeries, '-', 'LineWidth', 1.35);
    yline(0, ':');
    xlim([1 24]);
    xticks([1 6 12 18 24]);
    xlabel('Hour');
    ylabel('kW');
    title(panelTitle);
    grid on;
    if showLegend
        legend({'Pre-market net load', 'P2P traded energy', 'Residual grid exchange'}, 'Location', 'best');
    end
end
