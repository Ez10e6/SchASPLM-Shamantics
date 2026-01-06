
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, EVENT).

room_feature(FEATURE, ROOM).

precedence_constraint(EVENT1, EVENT2).

timeslot_constraint(EVENT, TIMESLOT).

% Generator

1 { assigned(Event, Room, Timeslot) : room(Room, _), timeslot(Timeslot) } 1 :- event(Event).


% Hard Constraints

% Students must not have overlapping events.
:- student(Student, Event1), student(Student, Event2), Event1 != Event2, timeslot_constraint(Event1, Timeslot1), timeslot_constraint(Event2, Timeslot2), Timeslot1 != Timeslot2.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- assigned(Event1, Room, Timeslot), assigned(Event2, Room, Timeslot), student(Student, Event1), student(Student, Event2), RoomFeature(FEATURE, Room), room_feature(FEATURE, Room), #count{ Student : student(Student, Event1), student(Student, Event2) } > Capacity, room(Room, Capacity).


% Rooms must have the required features for the event.
:- assigned(Event, Room, Timeslot), room_feature(FEATURE, Room), not feature_requirement(FEATURE, Event).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Room, Timeslot), assigned(Event2, Room, Timeslot), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Room, Timeslot), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(Event1, _, _), assigned(Event2, _, _), precedence_constraint(Event1, Event2), Event1 != Event2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslot",student(Student,Event),1) :- student(Student,Event), assigned(Event,_,Timeslot), Timeslot = #max{Timeslot}, room_feature("last_timeslot",Room), room(Room, _).


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots",student(Student,Event),1) :- student(Student,Event), 3 { timeslot_constraint(Event,Timeslot) } 3.


% Students should not have exactly one event in a day
penalty("EventOverlap", student(Student, Event), 1) :- student(Student, Event), 1 == #count{Event2 : student(Student, Event2)}.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
