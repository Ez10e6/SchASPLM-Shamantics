
nurse(1..4).
day(1..3).
shift_type("morning", 6); shift_type("midday", 8); shift_type("evening", 10).
:- schedule(N, D, S), not nurse_of(N, _), shift_type(S).
:- schedule(N, D, S), nurse_of(N, N1), S != "evening", D = 3, N1 != 2.
:- schedule(N, D, S), nurse_of(N, N1), S = "evening", D = 3, N1 != 1.
:- schedule(N, D, S), nurse_of(N, N1), S = "midday", D = 2, N1 != 3.
:- schedule(N, D, S), nurse_of(N, N1), S = "morning", D = 1, N1 != 4.
:- schedule(N, D, S), nurse_of(N, N1), S = "midday", D = 1, N1 != 4.
:- schedule(N, D, S), nurse_of(N, N1), S = "evening", D = 2, N1 != 3.
:- schedule(N, D, S), nurse_of(N, N1), S = "evening", D = 1, N1 != 4.
:- schedule(N, D, S), nurse_of(N, N1), S = "midday", D = 3, N1 != 2.
:- schedule(N, D, S), nurse_of(N, N1), S = "morning", D = 2, N1 != 3.
:- schedule(N, D, S), nurse_of(N, N1), S = "morning", D = 3, N1 != 2.
:- schedule(N, D, S), nurse_of(N, N1), S = "midday", D = 2, N1 != 4.
:- schedule(N, D, S), nurse_of(N, N1), S = "evening", D = 3, N1 != 2.
:- schedule(N, D, S), nurse_of(N, N1), S = "evening", D = 2, N1 != 4.
:- schedule(N, D, S), nurse_of(N, N1), S = "midday", D = 1, N1 != 3.
:- schedule(N, D, S) :- day(D), shift(S), not assigned(N, S, D).
:- schedule(N, D, S) :- day(D), shift(S), assigned(N, S, D), N != N1, N1 = #max{N2 : assigned(N2, S, D)}.
:- schedule(N, D, S) :- day(D), shift(S), assigned(N, S, D), N != N2, N2 = #min{N3 : assigned(N3, S, D)}.

% Generator

1 { scheduled(N, S, D) : shift(S), day(D) } 1 :- nurse(N).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- day(D), shift(S), #count{N : scheduled(N, S, D)} < Min.
:- day(D), shift(S), #count{N : scheduled(N, S, D)} > Max.


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(N), #count { D : scheduled(N, _, D) } < MinWork.
:- nurse(N), #count { D : scheduled(N, _, D) } > MaxWork.


% Nurses must have exactly 30 days of vacation.
:- #count {D : vacation(N, D)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- schedule(N, _, "night", _), schedule(N, _, "morning", D), D = #min {D1 : schedule(N, _, "afternoon", D1)}, D1 = D - 1.


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- schedule(N, _, "morning", D), schedule(N, _, "afternoon", D+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(N), #count { D : day(D), scheduled(N, "rest", D) } < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- schedule(N, _, "night", _), next_day(_, D), schedule(N, _, "night", D), special_rest(N, D).
:- schedule(N, _, "special_rest", _).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(N), #count{D: schedule(N, D, _)} < MinWork, min_work(MinWork).
:- nurse(N), #count{D: schedule(N, D, _)} > MaxWork, max_work(MaxWork).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
workload(N, D, S, V) :- schedule(N, D, S), V = 1.
workload(N, D, S, V) :- nurse(N), day(D), shift(S), V = #count{W : schedule(W, D, S), W != N}.
:- nurse(N), day(D), shift(S), workload(N, D, S, V), V != 2.
:- nurse(N), day(D), shift(S), workload(N, D, S, V), V > 2.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
