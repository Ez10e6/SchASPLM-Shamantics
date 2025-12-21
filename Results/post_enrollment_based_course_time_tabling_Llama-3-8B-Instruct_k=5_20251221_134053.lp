
% Instance Template
event(0..N_events-1).

room(Room, Capacity).

timeslot(Day, Hour).

student(Event).

feature(Event, Feature).

room_feature(Room, Feature).

precedence(Event1, Event2).

timeslot_constraint(Event, Timeslot).

% Generator

% 1
assigned(Timeslot, Room) :- room(Room, Capacity), timeslot(Timeslot).


% Hard Constraints

% Students must not have overlapping events.
:- assigned(Timeslot1, Room), assigned(Timeslot2, Room), Timeslot1 != Timeslot2.


% The amount of students in a room at the same time should not exceed the capacity of the room
Corrected Code:
:- assigned(Timeslot, R), #count{ S : student(S), assigned(Timeslot, R1), R1 != R } + #count{ S : student(S), assigned(Timeslot, R) } > Capacity, room(R, Capacity).


% Rooms must have the required features for the event.
:- assigned(Event, Timeslot), room_feature(Room, Feature), feature(Event, Feature), assigned(Timeslot, Room).


% There must be no more than one event in a room at a time.
:- assigned(Timeslot1, Room), assigned(Timeslot2, Room), Timeslot1 != Timeslot2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Room, Timeslot), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(Event1, Timeslot), assigned(Event2, Timeslot), precedence(Event1, Event2).




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslot", E, 1) :- event(E), timeslot_constraint(E, T), T = timeslot(_, H), H = 9.


% Students should not have events scheduled in three or more consecutive timeslots
has_event(Student,T) :- event(E), student(Student), timeslot_constraint(E,T).
consecutive_timeslots(T1,T2,T3) :- has_event(_,T1), has_event(_,T2), has_event(_,T3), T2 = T1 + 1, T3 = T2 + 1.
penalty("ConsecutiveEvents",Student,1) :- student(Student), consecutive_timeslots(T1,T2,T3), has_event(Student,T1), has_event(Student,T2), has_event(Student,T3).


% Students should not have exactly one event in a day
1 { assigned(Timeslot, Room) : student(Event), timeslot(Timeslot, Day), feature(Event, _) } 1 :- day(Day).
Corrected Code:
penalty("SingleEventPerDay", 1) :- Count = #count{E : assigned(T, R), timeslot(T, D), feature(E, _), student(E), day(D)}, Count = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
