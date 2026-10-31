
function critic = buildCriticNetwork(inputDim, hiddenUnits)
    layers = [
        featureInputLayer(inputDim, Normalization='none', Name='sa')
        fullyConnectedLayer(hiddenUnits, Name='fc1')
        reluLayer(Name='relu1')
        fullyConnectedLayer(hiddenUnits, Name='fc2')
        reluLayer(Name='relu2')
        fullyConnectedLayer(1, Name='qOut')
    ];
    critic = dlnetwork(layerGraph(layers));
end