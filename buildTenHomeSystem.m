% =========================================================================
% TEN-HOME DATA
% =========================================================================
function homes = buildTenHomeSystem(cfg)
    % Strong heterogeneity is introduced on purpose so that the hourly auction
    % sees simultaneous buyers and sellers. The 10 homes differ in:
    %   - daily demand shape and peak timing
    %   - PV magnitude and PV peak timing
    %   - EV arrival/departure schedules
    %   - initial EV and BESS states of charge

    loadEvening = [ ...
        1.20 1.15 1.10 1.05 1.05 1.15 1.35 1.65 1.90 2.05 2.10 2.00 ...
        1.90 1.95 2.10 2.35 2.80 3.50 4.20 4.45 4.10 3.60 2.70 1.85];

    loadDaytime = [ ...
        0.90 0.88 0.86 0.84 0.85 0.95 1.25 1.80 2.50 3.10 3.45 3.60 ...
        3.55 3.40 3.20 2.95 2.55 2.05 1.75 1.55 1.40 1.25 1.10 1.00];

    loadEarly = [ ...
        1.50 1.60 1.55 1.45 1.35 1.25 1.20 1.25 1.40 1.65 1.85 2.00 ...
        2.10 2.05 1.95 1.90 2.00 2.35 2.80 3.00 2.70 2.30 1.95 1.70];

    loadFlat = [ ...
        1.10 1.08 1.05 1.02 1.02 1.08 1.18 1.30 1.45 1.58 1.70 1.78 ...
        1.82 1.80 1.78 1.85 2.00 2.25 2.55 2.60 2.45 2.18 1.78 1.35];

    pvEast = [ ...
        0.00 0.00 0.00 0.00 0.05 0.20 0.65 1.45 2.55 3.60 4.35 4.70 ...
        4.50 3.85 2.95 1.85 0.90 0.25 0.00 0.00 0.00 0.00 0.00 0.00];

    pvSouth = [ ...
        0.00 0.00 0.00 0.00 0.00 0.10 0.35 0.95 1.85 2.95 4.00 4.75 ...
        5.00 4.70 4.05 3.05 1.95 0.95 0.25 0.00 0.00 0.00 0.00 0.00];

    pvWest = [ ...
        0.00 0.00 0.00 0.00 0.00 0.00 0.15 0.50 1.10 1.95 2.95 3.90 ...
        4.70 4.95 4.80 4.05 3.10 2.00 0.90 0.20 0.00 0.00 0.00 0.00];

    % Demand classes chosen to create simultaneous daytime buyers and sellers.
    demandClass = [1 2 3 4 1 2 3 4 1 2];
    pvClass =     [2 1 3 2 3 1 2 3 1 2];

    % Explicit profile shifts (hours) to stagger peaks.
    loadShiftHours = [-1 0 1 -2 2 -1 1 0 2 -2];
    pvShiftHours =   [ 0 -2 2 -1 1 -3 0 2 -2 1];

    loadScale = [1.15 0.92 1.28 1.00 1.08 0.88 1.22 0.96 1.12 0.90];
    pvScale   = [1.08 0.72 1.18 0.90 1.22 0.68 1.02 1.12 0.76 0.95];

    % Strong heterogeneity in EV schedules and initial SoC.
    evArrivalHours = [17 18 19 20 16 21 18 19 17 20];
    evDepartureHours = [6 7 8 9 7 6 8 9 7 8];
    initialEVSoC = [12 18 24 30 16 36 22 40 28 14];
    initialESSSoC = [2.4 3.2 5.0 7.8 4.1 8.7 3.5 6.4 9.0 5.6];

    for i = 1:cfg.numHomes
        switch demandClass(i)
            case 1
                loadBase = loadEvening;
            case 2
                loadBase = loadDaytime;
            case 3
                loadBase = loadEarly;
            otherwise
                loadBase = loadFlat;
        end

        switch pvClass(i)
            case 1
                pvBase = pvEast;
            case 2
                pvBase = pvSouth;
            otherwise
                pvBase = pvWest;
        end

        loadProfile = circshift(loadBase, [0, loadShiftHours(i)]);
        pvProfile = circshift(pvBase, [0, pvShiftHours(i)]);

        % Home-specific modulation to avoid synchronized shapes.
        h = 1:24;
        loadRipple = 0.18 * sin((h + 1.3 * i) / 24 * 2 * pi) + 0.09 * cos((h - 0.7 * i) / 24 * 4 * pi);
        pvRipple = 0.12 * sin((h + 0.9 * i) / 24 * 2 * pi);

        % Additional selective midday demand for some homes and stronger evening
        % peaks for others, to create overlap of deficits and surpluses.
        middayBump = zeros(1, 24);
        middayBump(10:15) = 0.35 * ((mod(i, 3) == 0) + 0.5 * (mod(i, 5) == 0));
        eveningBump = zeros(1, 24);
        eveningBump(18:22) = [0.10 0.22 0.32 0.20 0.10] * (0.6 + 0.1 * i);

        loadProfile = loadScale(i) * loadProfile + loadRipple + middayBump + 0.35 * (demandClass(i) == 1) * eveningBump;
        loadProfile = max(loadProfile, 0.55);

        pvProfile = pvScale(i) * pvProfile + pvRipple;
        pvProfile = min(max(pvProfile, 0.0), cfg.pvNorm);

        homes(i).name = sprintf('H%02d', i);
        homes(i).load = loadProfile;
        homes(i).pv = pvProfile;
        homes(i).evEta = 0.885 + 0.007 * mod(i - 1, 3);
        homes(i).essEta = 0.935 + 0.004 * mod(i - 1, 4);
        homes(i).initialEVSoC = initialEVSoC(i);
        homes(i).initialESSSoC = initialESSSoC(i);
        homes(i).evArrivalHour = evArrivalHours(i);
        homes(i).evDepartureHour = evDepartureHours(i);
    end
end

