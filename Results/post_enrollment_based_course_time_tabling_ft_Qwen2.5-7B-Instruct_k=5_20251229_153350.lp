
event(1..N).

room(1..M).

timeslot(1..D, 1..H).

student(1..K).

feature(1..F).

:- schedule(1, _), not topicOf(1, "Math").

:- schedule(2, _), not topicOf(2, "Math").

:- schedule(3, _), not topicOf(3, "Math").

:- schedule(4, _), not topicOf(4, "Math").

:- schedule(5, _), not topicOf(5, "Math").

:- schedule(6, _), not topicOf(6, "Math").

:- schedule(7, _), not topicOf(7, "Math").

:- schedule(8, _), not topicOf(8, "Math").

:- schedule(9, _), not topicOf(9, "Math").

:- schedule(10, _), not topicOf(10, "Math").

:- schedule(11, _), not topicOf(11, "Math").

:- schedule(12, _), not topicOf(12, "Math").

:- schedule(13, _), not topicOf(13, "Math").

:- schedule(14, _), not topicOf(14, "Math").

:- schedule(15, _), not topicOf(15, "Math").

:- schedule(16, _), not topicOf(16, "Math").

:- schedule(17, _), not topicOf(17, "Math").

:- schedule(18, _), not topicOf(18, "Math").

:- schedule(19, _), not topicOf(19, "Math").

:- schedule(20, _), not topicOf(20, "Math").

:- schedule(21, _), not topicOf(21, "Math").

:- schedule(22, _), not topicOf(22, "Math").

:- schedule(23, _), not topicOf(23, "Math").

:- schedule(24, _), not topicOf(24, "Math").

:- schedule(25, _), not topicOf(25, "Math").

:- schedule(26, _), not topicOf(26, "Math").

:- schedule(27, _), not topicOf(27, "Math").

:- schedule(28, _).

% Generator

:- schedule(Event, _, Room), topicOf(Event, "Math"), room(Room).


% Hard Constraints

% Students must not have overlapping events.
:- schedule(E1, T, R, D), schedule(E2, T, R, D),
topicOf(E1, "Math"), topicOf(E2, "Math"),
E1 != E2.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- scheduled(W, _, R, T), scheduled(V, _, R, T), W != V, topicOf(W, "Math"), topicOf(V, "Math"), R = 3, T = 3.


% Rooms must have the required features for the event.
:- schedule(Event, Room, _, _), requiredFeature(Event, Feature), not hasFeature(Room, Feature).


% There must be no more than one event in a room at a time.
:- schedule(E1, R, T), schedule(E2, R, T), E1 != E2.


% Events may only be scheduled in designated timeslots.
:- schedule(Event, _, Room, _), not timeslot(_, Room, _).


% Some events must be scheduled in a specific order.
:- schedule(E1, _, T, _), schedule(E2, _, T, _), E1 < E2, not edge(E1, E2).

edge(1, 2).
edge(2, 3).
edge(3, 4).




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 5, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 6, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 7, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 8, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 9, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 10, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 11, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 12, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 13, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 14, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 15, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 16, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 17, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 18, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 19, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 20, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 21, topicOf(Event, "Student").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Physics").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Chemistry").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Bio").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Art").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Music").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "PE").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "History").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Literature").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot = 22, topicOf(Event, "Math").
:- schedule(Event, Timeslot, _), timeslot(Timeslot), Timeslot


% Students should not have events scheduled in three or more consecutive timeslots
:- schedule(S, _, T), timeslot(T), T = #min{T1: schedule(S, _, T2), T2 = T1+1, T2 = T1+2}, topicOf(S, "Math").
:- #count{L: schedule(S, _, L)} >= 3.


% Students should not have exactly one event in a day
:- schedule(1, R, T), R = 1, topicOf(1, "Math"), R = 1, T = 1.
:- schedule(2, R, T), R = 1, topicOf(2, "Math"), R = 1, T = 2.
:- schedule(3, R, T), R = 1, topicOf(3, "Math"), R = 1, T = 3.
:- schedule(4, R, T), R = 1, topicOf(4, "Math"), R = 1, T = 4.
:- schedule(5, R, T), R = 1, topicOf(5, "Math"), R = 1, T = 5.
:- schedule(6, R, T), R = 1, topicOf(6, "Math"), R = 1, T = 6.
:- schedule(7, R, T), R = 1, topicOf(7, "Math"), R = 1, T = 7.
:- schedule(8, R, T), R = 1, topicOf(8, "Math"), R = 1, T = 8.
:- schedule(9, R, T), R = 1, topicOf(9, "Math"), R = 1, T = 9.
:- schedule(10, R, T), R = 1, topicOf(10, "Math"), R = 1, T = 1.
:- schedule(11, R, T), R = 1, topicOf(11, "Math"), R = 1, T = 2.
:- schedule(12, R, T), R = 1, topicOf(12, "Math"), R = 1, T = 3.
:- schedule(13, R, T), R = 1, topicOf(13, "Math"), R = 1, T = 4.
:- schedule(14, R, T), R = 1, topicOf(14, "Math"), R = 1, T = 5.
:- schedule(15, R, T), R = 1, topicOf(15, "Math"), R = 1, T = 6.
:- schedule(16, R, T), R = 1, topicOf(16, _).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
