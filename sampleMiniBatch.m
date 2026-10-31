function batch = sampleMiniBatch(buffer, batchSize)
    idx = randperm(buffer.size, batchSize);
    batch.state = buffer.state(:, idx);
    batch.action = buffer.action(:, idx);
    batch.reward = buffer.reward(:, idx);
    batch.nextState = buffer.nextState(:, idx);
    batch.done = buffer.done(:, idx);
end
