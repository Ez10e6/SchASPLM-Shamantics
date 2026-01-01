
:- scheduled(S, _, Sl), topicOf(W, "Networking"), S != "morning".

% Generator

:- schedule(S), not composite(V), not scheduled(_, _, _, D).


% Hard Constraints

% Students must not have overlapping events.
:- pulse_at("Relay",T): time(T), T<5.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- scheduled(P, _, _, V), topicOf(L, "AI").


% Rooms must have the required features for the event.
:- scheduled(P1, R, _), topicOf(W2, "Math"), topicOf(W2, "Math").


% There must be no more than one event in a room at a time.
:- scheduled(P1, R, _), topicOf(L1, "AI").


% Events may only be scheduled in designated timeslots.
:- scheduled(P1, _, T), topicOf(L2, "Math"), topicOf(S2, "Math").


% Some events must be scheduled in a specific order.
:- scheduled(P1, R, _), topicOf(W1, "Math").




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
:- scheduled(P1, _, T, _), scheduled(W2, R, _), topicOf(W2, "Math"), topicOf(L2, "Math").


% Students should not have events scheduled in three or more consecutive timeslots
room(1..0).


% Students should not have exactly one event in a day
:- refill(T), at(L1, T), V < 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
