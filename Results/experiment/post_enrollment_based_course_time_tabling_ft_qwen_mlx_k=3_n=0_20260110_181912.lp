
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, EVENT).

room_feature(FEATURE, ROOM).

precedence_constraint(EVENT1, EVENT2).

timeslot_constraint(EVENT, TIMESLOT).

% Generator

1 { assigned(Event, Timeslot, Room) : event(Event) } 1 :- room(Room, _), timeslot(Timeslot).


% Hard Constraints

% Students must not have overlapping events.
:- student(Student, Event1), student(Student, Event2), Event1 != Event2, timeslot(Timeslot1), timeslot(Timeslot2), assigned(Event1, Timeslot1, _), assigned(Event2, Timeslot2, _), Timeslot1 < Timeslot2, Timeslot1 + 1 = Timeslot2.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- room(Room, Capacity), #count{Event : assigned(Event, Timeslot, Room), timeslot_constraint(Event, Timeslot)} > Capacity.


% Rooms must have the required features for the event.
:- assigned(Event, Timeslot, Room), room_feature(FEATURE, Room), not feature_requirement(FEATURE, Event).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Timeslot, Room), assigned(Event2, Timeslot, Room), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Timeslot, Room), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(Event1, _, _, _), assigned(Event2, _, _, _), precedence_constraint(Event1, Event2), not assigned(Event1, Timeslot1, _, _), not assigned(Event2, Timeslot2, _, _), Timeslot1 < Timeslot2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslot",student(Student,Event),1) :- student(Student,Event), assigned(Event,Timeslot,Room), timeslot(Day,Hour), timeslot_constraint(Event,Timeslot), Hour = 8 + 4.


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots", Student, Penalty) :- student(Student, Event), 3 = #count{ Timeslot : assigned(Event, Timeslot), Timeslot2 : assigned(Event, Timeslot2), Timeslot3 : assigned(Event, Timeslot3), Timeslot < Timeslot2, Timeslot2 < Timeslot3, Timeslot < Timeslot3 }.


% Students should not have exactly one event in a day
penalty("SingleEventPerDay", student(Student), 1) :- #count{Event : assigned(Event, Timeslot, Room), student(Student), timeslot(Day, _), timeslot_constraint(Event, Timeslot)} = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
