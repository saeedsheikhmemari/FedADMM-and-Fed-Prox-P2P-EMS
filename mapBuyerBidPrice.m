function bidPrice = mapBuyerBidPrice(actionValue, gridSell, gridBuy)
    frac = 0.5 * (actionValue + 1);    % [0,1]
    mid = 0.5 * (gridBuy + gridSell);
    bidPrice = mid + frac * (gridBuy - mid);
    bidPrice = min(max(bidPrice, mid), gridBuy);
end