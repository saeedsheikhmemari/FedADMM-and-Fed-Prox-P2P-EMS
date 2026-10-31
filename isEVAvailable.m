function tf = isEVAvailable(hour, arrivalHour, departureHour)
    if arrivalHour > departureHour
        tf = (hour >= arrivalHour) || (hour < departureHour);
    elseif arrivalHour < departureHour
        tf = (hour >= arrivalHour) && (hour < departureHour);
    else
        tf = true;
    end
end