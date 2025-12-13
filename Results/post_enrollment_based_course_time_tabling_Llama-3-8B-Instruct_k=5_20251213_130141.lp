
% Instance Template
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Event).

feature(Event, Feature).

room_feature(Room, Feature).

precedence_event(Event1, Event2).

timeslot_event(Event, Timeslot).

% Generator

% 1
Corrected code:
1 { assigned(Event, Timeslot, Room) : room(Room,Capacity) } :- event(Event), timeslot(Timeslot).


% Hard Constraints

% Students must not have overlapping events.
:- assigned(Event1, Timeslot1, Room), assigned(Event2, Timeslot1, Room), Event1!= Event2.


% The amount of students in a room at the same time should not exceed the capacity of the room
Corrected code:
room_capacity_exceeded(Room) :- room(Room, Capacity), #count{ Student : student(Student, Event), assigned(Event, Timeslot, Room), timeslot_event(Event, Timeslot) } > Capacity.


% Rooms must have the required features for the event.
:- assigned(Event, Timeslot, Room), feature(Event, Feature), not room_feature(Room, Feature).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Timeslot, Room), assigned(Event2, Timeslot, Room), Event1!= Event2.


% Events may only be scheduled in designated timeslots.
Corrected code:
assigned(Event, Timeslot, Room) :- event(Event), timeslot(Timeslot), timeslot_event(Event, Timeslot), room(Room,Capacity).


% Some events must be scheduled in a specific order.
:- precedence_event(Event1, Event2), timeslot_event(Event1, Timeslot1), timeslot_event(Event2, Timeslot2), Timeslot1!= Timeslot2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslot",student(Event),1) :- event(Event), timeslot_event(Event,Timeslot), timeslot(Timeslot,Day,Hour), Hour = 9.


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveEvents",consecutive_events(Student),1) :- student(Student), timeslot_event(Event,Timeslot), timeslot_event(Event,Timeslot+1), timeslot_event(Event,Timeslot+2), assigned(Event,Timeslot,_), assigned(Event,Timeslot+1,_), assigned(Event,Timeslot+2,_), student(Student,_).


% Students should not have exactly one event in a day
student_event_count(Count, Student, Day) :- student(Student), timeslot(Day, _), Count = #count{ Event : event(Event), student(Event, Student), timeslot_event(Event, Timeslot), timeslot(Timeslot, Day) }.
penalty("SingleEvent", student_event_count(Count, Student, Day), (1-Count)*1) :- student_event_count(Count, Student, Day), Count = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
