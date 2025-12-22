
% Instance Template
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, Event).

room_feature(FEATURE, Room).

precedence_constraint(Event1, Event2).

timeslot_constraint(Event, Timeslot).

% Generator

% 1
1 { event(Event) } 1.
1 { assign(Event, Room, Timeslot) : room(Room), timeslot(Timeslot) } 1 :- event(Event).


% Hard Constraints

% Students must not have overlapping events.
:- student(Student), event(Event1), event(Event2), precedence_constraint(Event1, Event2), timeslot_constraint(Student, Event1, Timeslot1), timeslot_constraint(Student, Event2, Timeslot2), Timeslot1 < Timeslot2, overlap(Timeslot1, Timeslot2).


% The amount of students in a room at the same time should not exceed the capacity of the room
:- assign(Event1, Room), assign(Event2, Room), event(Event1), event(Event2), #count{Student : student(Student, Event1), student(Student, Event2)} > Capacity, room(Room, Capacity).


% Rooms must have the required features for the event.
:- assign(Event, Room, Timeslot), room_feature(FEATURE, Room), not feature_requirement(FEATURE, Event).


% There must be no more than one event in a room at a time.
:- assign(Event1, Room, Timeslot), assign(Event2, Room, Timeslot), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- event(Event), timeslot(Timeslot), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- event(Event1), event(Event2), precedence_constraint(Event1, Event2), assign(Event1, Room1, Timeslot1), assign(Event2, Room2, Timeslot2), Timeslot1 > Timeslot2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslotConstraint", student(Student, Event), 1) :-
    student(Student, Event),
    timeslot(Day, Hour),
    assign(Event, _, _),
    Hour = 8 + 6 * (Day - 1),
    Day =< 5.


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots", Student, 1) :-
3 #<= #count{ Timeslot : assign(Student, _, Timeslot), assign(Student, _, Timeslot+1), assign(Student, _, Timeslot+2) }.


% Students should not have exactly one event in a day
penalty("SingleEventPerDay", single_event_per_day(Student, Day), 1) :-
#count{Timeslot : assign(Event, _, Timeslot), event(Event), student(Student), timeslot(Day, _)} = 1,
student(Student),
timeslot(Day, _).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
