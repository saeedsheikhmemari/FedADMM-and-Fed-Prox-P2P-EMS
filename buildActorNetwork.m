function actor = buildActorNetwork(stateDim, actionDim, hiddenUnits)
    layers = [
        featureInputLayer(stateDim, Normalization='none', Name='state')
        fullyConnectedLayer(hiddenUnits, Name='fc1')
        reluLayer(Name='relu1')
        fullyConnectedLayer(hiddenUnits, Name='fc2')
        reluLayer(Name='relu2')
        fullyConnectedLayer(2 * actionDim, Name='policyOut')
    ];
    actor = dlnetwork(layerGraph(layers));
end