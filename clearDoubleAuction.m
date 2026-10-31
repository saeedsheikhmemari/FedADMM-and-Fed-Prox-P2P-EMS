
function market = clearDoubleAuction(desiredBidQty, bidPrice, gridBuy, gridSell, cfg)
    numHomes = numel(desiredBidQty);
    qP2P = zeros(numHomes, 1);

    buyerIdx = find(desiredBidQty > 1e-9);
    sellerIdx = find(desiredBidQty < -1e-9);

    buyQty = desiredBidQty(buyerIdx);
    sellQty = -desiredBidQty(sellerIdx);
    buyPrice = bidPrice(buyerIdx);
    sellPrice = bidPrice(sellerIdx);

    [buyPriceSorted, buyOrder] = sort(buyPrice, 'descend');
    [sellPriceSorted, sellOrder] = sort(sellPrice, 'ascend');

    buyerIdx = buyerIdx(buyOrder);
    sellerIdx = sellerIdx(sellOrder);
    buyQty = buyQty(buyOrder);
    sellQty = sellQty(sellOrder);

    bi = 1;
    si = 1;
    tradedValue = 0;
    totalVolume = 0;

    while bi <= numel(buyerIdx) && si <= numel(sellerIdx)
        if buyPriceSorted(bi) + 1e-9 < sellPriceSorted(si)
            break;
        end

        tradedQty = min(buyQty(bi), sellQty(si));
        if tradedQty <= 0
            break;
        end

        clearingPrice = 0.5 * (buyPriceSorted(bi) + sellPriceSorted(si));

        qP2P(buyerIdx(bi)) = qP2P(buyerIdx(bi)) + tradedQty;
        qP2P(sellerIdx(si)) = qP2P(sellerIdx(si)) - tradedQty;

        tradedValue = tradedValue + clearingPrice * tradedQty;
        totalVolume = totalVolume + tradedQty;

        buyQty(bi) = buyQty(bi) - tradedQty;
        sellQty(si) = sellQty(si) - tradedQty;

        if buyQty(bi) <= 1e-9
            bi = bi + 1;
        end
        if sellQty(si) <= 1e-9
            si = si + 1;
        end
    end

    if totalVolume > 0
        localPrice = tradedValue / totalVolume;
    else
        localPrice = cfg.marketPriceFallbackFrac * (gridBuy + gridSell);
    end

    market.localPrice = localPrice;
    market.qP2P = qP2P;
    market.totalVolume = totalVolume;
end
