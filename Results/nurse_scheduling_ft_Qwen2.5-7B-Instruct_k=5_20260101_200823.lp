
:- #sum {X : sum(X)} != n.

% Generator

{ occurs(s, S, T) : edge(V) } 1 :- not edge(L).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- queens(R, C), queens(S, D), R!=S.


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
user(1..1).


% Nurses must have exactly 30 days of vacation.
loc("morning"; "afternoon"; "midday", 100; "Litero", 150).


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
topic("AI"; "Bio"; "Math"; "Chem").
:- at("NodeA",T), at("NodeX",T+1), T < 15.


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- scheduled(_, R, _, _), topicOf(W, "Networking"), topicOf(S, "AI").


% Each nurse has at least two ordinary rest days for every window of fourteen days
{ occurs(some_action,T) } :- step(T), num(Y).


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
topic("Networking"; "Bio"; "Math").
:- refill(T), at(L,N).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
user(1..1).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
{ occurs(some_action,T) } :- step(T).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
