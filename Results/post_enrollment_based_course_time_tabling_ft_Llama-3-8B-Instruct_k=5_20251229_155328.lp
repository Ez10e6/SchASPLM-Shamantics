
event(0..N_events-1).
room(Room, Capacity).
timeslot(D, H).
student(Event).
feature(Feature, Event).
room_feature(Feature, Room).
:- scheduled(1, R, Sl, D, H), scheduled(2, R, Sl, D, H), event1 = 1, event2 = 2, event1 != event2.
:- scheduled(1, R, Sl, D, H), timeslot(1, D, H), timeslot(2, D, H), event1 = 1, event2 = 2, event1 != event2.

% Generator

event(0..N_events-1).
room(Room, Capacity).
timeslot(D, H).
:- scheduled(Event, R, Sl, D, H), room(R), timeslot(Sl, D, H).


% Hard Constraints

% Students must not have overlapping events.
:- scheduled(1, R, Sl, D, H), scheduled(2, R, Sl, D, H), event1 = 1, event2 = 2, event1 != event2.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- scheduled(Event, R, Sl, D, H), #count{Event : scheduled(Event, R, Sl, D, H)} > Room(C), room(R, Capacity).


% Rooms must have the required features for the event.
:- scheduled(Event, R, Sl, D, H), feature(Feature, Event), -room_feature(Feature, R).


% There must be no more than one event in a room at a time.
:- scheduled(Event1, R, Sl, D, H), scheduled(Event2, R, Sl, D, H), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- scheduled(Event, R, Sl, D, H), timeslot(Sl, D, H), not room(R, Sl, D, H).


% Some events must be scheduled in a specific order.
:- scheduled(Event1, _, Sl, D, H), scheduled(Event2, _, Sl, D, H), Event1 != Event2, Event1 > Event2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
:- scheduled(Event, R, Sl, D, H), Sl = 9, timeslot(Sl, D, H), student(Event).


% Students should not have events scheduled in three or more consecutive timeslots
:- scheduled(Stu, R, Sl, D, H), scheduled(Stu, R, Sl+1, D, H), scheduled(Stu, R, Sl+2, D, H), Stu = 1.


% Students should not have exactly one event in a day
event(0..N_events-1).
room(Room, Capacity).
timeslot(D, H).
student(Event).
feature(Feature, Event).
room_feature(Feature, Room).
:- scheduled(1, R, Sl, D, H), scheduled(2, R, Sl, D, H), event1 = 1, event2 = 2, event1 != event2.
:- scheduled(1, R, Sl, D, H), timeslot(1, D, H), timeslot(2, D, H), event1 = 1, event2 = 2, event1 != event2.
event(0..N_events-1).
room(Room, Capacity).
timeslot(D, H).
:- scheduled(Event, R, Sl, D, H), room(R), timeslot(Sl, D, H).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
