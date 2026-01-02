
% Instance Template
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, Event).

room_feature(FEATURE, Room).

precedence_constraint(EVENT1, EVENT2).

timeslot_constraint(EVENT, TIMESLOT).

% Generator

% Instance Template
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, Event).

room_feature(FEATURE, Room).

precedence_constraint(EVENT1, EVENT2).

timeslot_constraint(EVENT, TIMESLOT).

% Generator
{ assigned(Event, Room, Timeslot) : room(Room,_), timeslot(Timeslot) } 1 :- event(Event).


% Hard Constraints

% Students must not have overlapping events.
:- assigned(Event1, _, Timeslot1), assigned(Event2, _, Timeslot2), student(Student, Event1), student(Student, Event2), Timeslot1 != Timeslot2, overlap(Timeslot1, Timeslot2).


% The amount of students in a room at the same time should not exceed the capacity of the room
:- room(Room, Capacity), room_feature(capacity, Room), assigned(Event1, Room, Timeslot), assigned(Event2, Room, Timeslot), student(Student, Event1), student(Student, Event2), Event1 != Event2, #count{Student : student(Student, Event1), student(Student, Event2)} > Capacity.


% Rooms must have the required features for the event.
:- assigned(Event, Room, Timeslot), room_feature(FEATURE, Room), \+ feature_requirement(FEATURE, Event).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Room, Timeslot), assigned(Event2, Room, Timeslot), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Room, Timeslot), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(EVENT1, _, _, _), assigned(EVENT2, _, _, _), precedence_constraint(EVENT1, EVENT2), assigned(EVENT2, _, _, _), assigned(EVENT1, _, _, _).




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslotConstraint",student(Student,Event),1) :- assigned(Event,_,(Day,4)), student(Student,Event).


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots",student(Student,Event),(N-1)*1) :-
student(Student,Event),
assigned(Event,_,Timeslot1),
assigned(Event,_,Timeslot2),
assigned(Event,_,Timeslot3),
Timeslot2 = Timeslot1 + 1,
Timeslot3 = Timeslot1 + 2,
count_consecutive(Timeslot1,Timeslot2,Timeslot3,Student,N).


% Students should not have exactly one event in a day
penalty("SingleEventPerDay", single_event_per_day(Student, Day), 1) :-
#count{Event : assigned(Event, _, (Day, _)), student(Student), timeslot(_, _) } = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
