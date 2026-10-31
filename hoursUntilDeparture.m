function dt = hoursUntilDeparture(currentHour, departureHour)
    if currentHour <= departureHour
        dt = departureHour - currentHour;
    else
        dt = (24 - currentHour) + departureHour;
    end
end

