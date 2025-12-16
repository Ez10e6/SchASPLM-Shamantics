
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

% 1
1 { event(Event) } 1.
1 { assign(Event, Timeslot, Room) : room(Room), timeslot(Timeslot) } 1.


% Hard Constraints

% Students must not have overlapping events.
:- student(Student), event(Event1), event(Event2), precedence_constraint(Event1, Event2), timeslot(Timeslot1), timeslot(Timeslot2), Timeslot1 < Timeslot2, assign(Event1, Timeslot1, _), assign(Event2, Timeslot2, _).


% The amount of students in a room at the same time should not exceed the capacity of the room
:- room(Room, Capacity), #count{Event : assign(Event, Timeslot, Room), event(Event)} > Capacity.


% Rooms must have the required features for the event.
:- assign(Event, Timeslot, Room), room_feature(FEATURE, Room), \+ feature_requirement(FEATURE, Event).


% There must be no more than one event in a room at a time.
:- assign(Event1, Timeslot, Room), assign(Event2, Timeslot, Room), Event1!= Event2.


% Events may only be scheduled in designated timeslots.
:- event(Event), timeslot(Timeslot), not assign(Event, Timeslot, _).


% Some events must be scheduled in a specific order.
:- event(Event1), event(Event2), precedence_constraint(Event1, Event2), assign(Event1, Timeslot1, Room1), assign(Event2, Timeslot2, Room2), Timeslot1 >= Timeslot2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslotConstraint", student(Student, Event), 1) :- student(Student, Event), timeslot(Day, Hour), assign(Event, Timeslot, Room), Timeslot = { Day, 8 }, Day in 1..5.


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots", student(Student), N) :-
    N #>= 3,
    #count{ Timeslot : timeslot_constraint(Student, Timeslot) } #= N,
    N #> 0.


% Students should not have exactly one event in a day
penalty("SingleEventPerDay", single_event_per_student(Student, Day), 1) :-
#count{Timeslot : assign(Student, Timeslot, _), timeslot(Day, Timeslot)} = 1,
student(Student),
timeslot(Day, _).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
