
% Instance Template
event(0 .. N_events-1).

room(Room, Capacity).

timeslot(Day, Hour).

student(Event).

feature(Event, Feature).

room_feature(Room, Feature).

precedence(Event1, Event2).

timeslot_constraint(Event, Timeslot).

% Generator

% 1
1 { assigned(Event, Timeslot, Room) : room(Room,_) } 1 :- event(Event).


% Hard Constraints

% Students must not have overlapping events.
:- assigned(Event1, Timeslot1, Room), assigned(Event2, Timeslot1, Room), Event1 != Event2.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- room(Room, Capacity), #count{Student : student(Student, _), assigned(_, _, Room)} > Capacity.


% Rooms must have the required features for the event.
:- assigned(Event, Timeslot, Room), room_feature(Room, Feature), not feature(Event, Feature).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Timeslot, Room), assigned(Event2, Timeslot, Room), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Timeslot, Room), timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(Event1, _, _), assigned(Event2, _, _), precedence(Event1, Event2).




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslot", student(Event), 1) :- event(Event), assigned(Event, Timeslot, Room), timeslot(Day, Hour), Timeslot = (Day, Hour), Hour = 9.


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots", student(Event), 1) :- event(Event), assigned(Event, T1, _), assigned(Event, T2, _), assigned(Event, T3, _), T1 < T2, T2 < T3.


% Students should not have exactly one event in a day
student_event_count(Count, Student, Day) :- student(Student), #count{Event : event(Event), assigned(Event, _, _, Day), student(Event, Student)} = Count.
penalty("SingleEvent", student_event_count(Count, Student, Day), Count*1) :- student_event_count(Count, Student, Day), Count = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
