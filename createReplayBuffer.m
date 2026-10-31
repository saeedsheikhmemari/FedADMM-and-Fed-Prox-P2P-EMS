% =========================================================================
% REPLAY BUFFER
% =========================================================================
function buffer = createReplayBuffer(capacity, stateDim, actionDim)
    buffer.state = zeros(stateDim, capacity);
    buffer.action = zeros(actionDim, capacity);
    buffer.reward = zeros(1, capacity);
    buffer.nextState = zeros(stateDim, capacity);
    buffer.done = zeros(1, capacity);

    buffer.capacity = capacity;
    buffer.ptr = 1;
    buffer.size = 0;
end