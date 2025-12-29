
team(1..4).
venues(1..3).
dates(1..6).

% Generator

:- schedule(1, T1, V, D), schedule(2, T2, V, D), T1 != T2.


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- schedule(1, T1, _, D), schedule(2, T2, _, D), T1 != T2, game(T1, T2).


% Every team plays every other team at least once.
:- schedule(T1, _, V, D), schedule(T2, _, V, D), T1 != T2, not played(T1, T2).


% No team plays itself.
:- schedule(T, _, _, _), T = 1, T = 2.




% Soft Constraints

% Teams should play at the same venue as little as possible.
used_venue(N, V) :- schedule(_, _, V, _), N = #count {T : selected_team(T), game(T, V)}.
#minimize {N : used_venue(N, V)}.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
