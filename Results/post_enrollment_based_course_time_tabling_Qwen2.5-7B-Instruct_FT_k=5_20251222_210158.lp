
% Events: A set of N events.
event(0..N-1).

% Rooms: A set of rooms in which the events can be scheduled.
room(0..M-1, Capacity).

% Timeslots: A set of timeslots in which the events can be scheduled.
timeslot(0..P-1, Day, Hour).

% Students: A set of students that are attending a set of events.
student(0..Q-1).

% Feature Requirements: A set of features which events may have.
feature(0..R-1).

% Room Features: A set of features which rooms may have.
room_feature(F, R) :- feature(F), room(R, _).

% Precedence Constraints: A set of constraints that specify if some events must be scheduled before other events.
:- schedule_event(E1, T1), schedule_event(E2, T2), E1 < E2 < T1 <= T2.

% Timeslot Constraints: A set of constraints for each event that specify the timeslots in which the event can be scheduled.
:- schedule_event(E, T), not timeslot_of(E, T).

timeslot_of(E, T) :- event(E), timeslot(T, _, _).

% Generator

% Events: A set of N events.
event(0..N-1).

% Rooms: A set of rooms in which the events can be scheduled.
room(0..M-1).

% Timeslots: A set of timeslots in which the events can be scheduled.
timeslot(0..P-1, Day, Hour).

% Each event must be scheduled once.
% There is exactly one schedule_event per event.
1 { schedule_event(E, T, R) : room(R), timeslot(T, _, _) } 1 :- event(E).


% Hard Constraints

% Students must not have overlapping events.
:- schedule_event(E1, T1, _), schedule_event(E2, T2, _), student(S),
T1 = T2, Hour1 < Hour2,
enrolled(S, E1), enrolled(S, E2).


% The amount of students in a room at the same time should not exceed the capacity of the room
:- schedule_event(E1, T, R, S), schedule_event(E2, T, R, S), E1 != E2, room(R, Capacity), S + 1 > Capacity.


% Rooms must have the required features for the event.
:- schedule_event(E, T, R), has_feature(E, F), not room_feature(F, R).


% There must be no more than one event in a room at a time.
:- schedule_event(E1, T, R), schedule_event(E2, T, R), E1 != E2.


% Events may only be scheduled in designated timeslots.
:- schedule_event(E, T, R), not timeslot(T, _, _).


% Some events must be scheduled in a specific order.
:- schedule_event(E1, T1), schedule_event(E2, T2), E1 < E2 < T1 <= T2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
:- schedule_event(E, T, R), student(E), timeslot(T, Day, 8), T = P-1.
penalty("LastSlot",E,1) :- schedule_event(E, T, _), student(E), timeslot(T, _, 8), T = #max{T1: timeslot(T1, _, 8)}.


% Students should not have events scheduled in three or more consecutive timeslots
consecutive_timeslots(T, H, R) :- timeslot(T, _, H), timeslot(T+1, _, H), timeslot(T+2, _, H).
:- schedule_event(E, T, _), schedule_event(E, T+1, _), schedule_event(E, T+2, _), consecutive_timeslots(T, H, R).
penalty("ConsecutiveTimeslots",E,1) :- schedule_event(E, T, _), schedule_event(E, T+1, _), schedule_event(E, T+2, _), consecutive_timeslots(T, H, R).


% Students should not have exactly one event in a day
has_event(E, S, T, R) :- schedule_event(E, T, R), student(S), E = S.
:- student(S), day(D), #count{E : has_event(E, S, T, D)} = 1.

penalty("SoloDay", has_event(E, S, T, D), 1) :- student(S), day(D), #count{E : has_event(E, S, T, D)} = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
