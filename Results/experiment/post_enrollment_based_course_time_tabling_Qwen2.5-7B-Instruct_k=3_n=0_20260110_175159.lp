

event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, Event).

room_feature(FEATURE, Room).

precedence_constraint(EVENT1, EVENT2).

timeslot_constraint(EVENT, TIMESLOT).

% Generator

1 { assigned(Event, Timeslot, Room) : room(Room) } 1 :- event(Event).

room(Room) :- assigned(_, _, Room).


% Hard Constraints

% Students must not have overlapping events.
:- assigned(student(Student1, Event1), Timeslot1, Room1), assigned(student(Student1, Event2), Timeslot2, Room2), Timeslot1 != Timeslot2, Day1 = Day2.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- assigned(Event1, Timeslot, Room), assigned(Event2, Timeslot, Room), student(Student, Event1), student(Student, Event2), room(Room, Capacity), #count{Student: student(Student, Event1), student(Student, Event2)} > Capacity.


% Rooms must have the required features for the event.
:- assigned(Event, Timeslot, Room), room_feature(FEATURE, Room), \+ feature_requirement(FEATURE, Event).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Timeslot, Room), assigned(Event2, Timeslot, Room), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Timeslot, Room), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(EVENT1, TIMESLOT1, _), assigned(EVENT2, TIMESLOT2, _), precedence_constraint(EVENT1, EVENT2), TIMESLOT1 > TIMESLOT2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslotConstraint",student(Student,Event),1) :- assigned(Event,Timeslot,Room), student(Student,Event), Timeslot = (Day,4), room(Room).


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots",student(Student,Event),(N-1)*1) :-
student(Student,Event),
3 { timeslot(Day,Hour) : assigned(Event,Day,Hour) } 3,
N = { timeslot(Day,Hour) : assigned(Event,Day,Hour) }.


% Students should not have exactly one event in a day
penalty("SingleEventPerDay", single_event_per_day(Student, Day), 1) :-
#count{Event : assigned(Student, Timeslot, Room), timeslot(Day, _), student(Student)} = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
