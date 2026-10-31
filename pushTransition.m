function buffer = pushTransition(buffer, state, action, reward, nextState, doneFlag)
    idx = buffer.ptr;

    buffer.state(:, idx) = state;
    buffer.action(:, idx) = action;
    buffer.reward(1, idx) = reward;
    buffer.nextState(:, idx) = nextState;
    buffer.done(1, idx) = double(doneFlag);

    buffer.ptr = idx + 1;
    if buffer.ptr > buffer.capacity
        buffer.ptr = 1;
    end
    buffer.size = min(buffer.size + 1, buffer.capacity);
end