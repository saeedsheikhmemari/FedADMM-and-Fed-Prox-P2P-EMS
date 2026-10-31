function askPrice = mapSellerAskPrice(actionValue, gridSell, gridBuy)
    frac = 0.5 * (actionValue + 1);    % [0,1]
    mid = 0.5 * (gridBuy + gridSell);
    askPrice = gridSell + frac * (mid - gridSell);
    askPrice = min(max(askPrice, gridSell), mid);
end