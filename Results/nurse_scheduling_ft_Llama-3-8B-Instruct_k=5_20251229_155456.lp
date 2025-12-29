
nurse(N1 .. N2).
day(0 .. N3-1).
shift_type(Shift_type, Duration).
room(Shift_type, Min, Max).
room(Shift_type, Min, Max, Preferred).
min(N, Min, Max).
max(N, Min, Max).
min(Shift_type, Min, Max, Preferred).
max(Shift_type, Min, Max, Preferred).

% Generator

assigned(N, S, D) :- shift(S), day(D), N = #count{N1 : shift(S1), day(D1), N1 != N}.


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift(Shift), day(D), #count{N : assigned(N, Shift, D)} < Min, shift(Shift), day(D), #count{N : assigned(N, Shift, D)} > Max.


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- #min{H, N: assigned(N, _, H)} > MinH, #min{H, N: assigned(N, _, H)} < MaxH.


% Nurses must have exactly 30 days of vacation.
:- assigned(_, "vacation", _), #count{D : assigned(_, "vacation", D)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(N, S, D), assigned(N, S1, D+1), S = "night", S1 != "morning", S1 != "afternoon".


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(N, "afternoon", D), assigned(N, "morning", D+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- day(D), #count{D1: assigned(_, S, D1), shift(S), S != "rest", S != "special_rest", S != "vacation", D1 < D, D1 < D + 14} >= 2, day(D), #count{D1: assigned(_, S, D1), shift(S), S = "rest", S = "special_rest", S = "vacation", D1 < D, D1 < D + 14} <= 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(N, Shift, D), Shift = night, assigned(N, Shift, D+1), Shift = night, Shift != special_rest.


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- #min{H, N: assigned(N, _, H)} > MinH, min(N, MinH, MaxH).
:- #max{H, N: assigned(N, _, H)} < MaxH, min(N, MinH, MaxH).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift(N, S, D) :- assigned(N, S, D).
shift_count(N, S, D) :- shift(N, S, D).
preferred_shift(N, S, D) :- room(S, _, P), shift(N, S, D), P = #count{S1 : shift(N1, S1, D), N1 != N}.
penalty("ShiftPreference",shift(N, S, D), (P-ShiftCount)*1) :- shift(N, S, D), preferred_shift(N, S, D), ShiftCount = #count{S1 : shift(N1, S, D), N1 != N}.
penalty("ShiftPreference",shift(N, S, D), (ShiftCount-P)*1) :- shift(N, S, D), preferred_shift(N, S, D), ShiftCount = #count{S1 : shift(N1, S, D), N1 != N}.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
