

event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student, Event).

feature_requirement(FEATURE, Event).

room_feature(FEATURE, Room).

precedence_constraint(EVENT1, EVENT2).

timeslot_constraint(EVENT, TIMESLOT).

% Generator

{ assigned(Event, Timeslot, Room) : timeslot(Timeslot), room(Room) } 1 :- event(Event).



% Hard Constraints

% Students must not have overlapping events.
:- assigned(student(Student1, Event1), Timeslot1, Room1), assigned(student(Student1, Event1), Timeslot2, Room2), overlap(Timeslot1, Timeslot2).


% The amount of students in a room at the same time should not exceed the capacity of the room
:- room(Room, Capacity), event(Event),
#count{Student : student(Student, Event), assigned(Event, Timeslot, Room), timeslot(Timeslot)} > Capacity.


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
penalty("LastTimeslotConstraint",student(Student,Event),(1*1)) :- student(Student,Event), assigned(Event,Timeslot,Room), timeslot(Day,Hour), Timeslot = timeslot(Day,5).


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots", student(Student, Event), (N-1)*1) :-
    student(Student, Event),
    { assigned(Event, Timeslot, Room) : timeslot(Timeslot) } #>= 3,
    N = { Timeslot1, Timeslot2, Timeslot3 :
        assigned(Event, Timeslot1, Room),
        assigned(Event, Timeslot2, Room),
        assigned(Event, Timeslot3, Room),
        Timeslot2 = Timeslot1 + 1,
        Timeslot3 = Timeslot2 + 1
    }.


% Students should not have exactly one event in a day
To correctly enforce the intended semantics, we need to count the number of events each student has on each day and apply a penalty if the count is exactly one. Here is the corrected ASP code:

penalty("SingleEventPerDay", Student, 1) :-
    #count{Event : assigned(Student, Event, Timeslot, Room), timeslot(Timeslot, Day)} = 1,
    student(Student),
    timeslot(_, Day).

This rule checks for each student and each day whether they have exactly one event assigned. If so, it applies a penalty of 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
