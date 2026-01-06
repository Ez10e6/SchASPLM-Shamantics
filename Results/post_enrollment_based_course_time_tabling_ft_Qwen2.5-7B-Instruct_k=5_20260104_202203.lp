
% Instance Template
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, EVENT).

room_feature(FEATURE, ROOM).

precedence_constraint(EVENT1, EVENT2).

timeslot_constraint(EVENT, TIMESLOT).

% Generator

1 { assigned(Event, Timeslot, Room) : room(Room), timeslot(Timeslot) } 1 :- event(Event).


% Hard Constraints

% Students must not have overlapping events.
:- student(Student), event(Event1), event(Event2), precedence_constraint(Event1, Event2), assigned(Event1, Timeslot1, Room1), assigned(Event2, Timeslot2, Room2), D1 != D2, D1 = D2 + 1, D1 = D2 - 1.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- assigned(Event1, Timeslot, Room), assigned(Event2, Timeslot, Room), student(Student, Event1), student(Student, Event2), room(Room, Capacity), #count{Event : assigned(Event, Timeslot, Room), student(Student, Event)} > Capacity.


% Rooms must have the required features for the event.
:- assigned(Event, Timeslot, Room), room_feature(FEATURE, Room), not feature_requirement(FEATURE, Event).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Timeslot, Room), assigned(Event2, Timeslot, Room), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Timeslot, Room), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(Event1, Timeslot1, Room), assigned(Event2, Timeslot2, Room), precedence_constraint(Event1, Event2), Timeslot1 > Timeslot2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslot",student(Student,Event),1) :- assigned(Event,Timeslot,Room), student(Student,Event), timeslot(Day,Hour), last_timeslot(Timeslot).


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots",student(Student,Event),1) :- student(Student,Event), 3 { assigned(Event,Timeslot,_) : timeslot(Timeslot) } 3.


% Students should not have exactly one event in a day
penalty("OneEventPerDay", student(Student), 1) :- #count{Event : assigned(Event, Timeslot, Room), student(Student), timeslot(Day, _), timeslot(Timeslot, Day)} = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
