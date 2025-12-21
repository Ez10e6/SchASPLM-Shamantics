
% Events: A set of 5 events.
event(1..5).

% Rooms: A set of 3 rooms.
room(1..3).

% Timeslots: A set of 4 timeslots per day over two days.
timeslot(0..7).

% Students: All events are attended by the same set of 4 students.
student(1..4).

% Feature Requirements: Events 1 and 2 must be in rooms with projector; Event 3 must be in a room with whiteboard.
feature("projector", 1).
feature("whiteboard", 3).

% Room Features: Room 1 has projector; Room 2 has whiteboard; Room 3 has both.
has_feature("projector", 1).
has_feature("whiteboard", 2).
has_feature("projector", 3).
has_feature("whiteboard", 3).

% Precedence Constraints: Event 1 must be on Day 1; Event 2 must be after Event 1; Event 3 must be on Day 2; Event 4 must be on Day 1 and in timeslot 2 or 3.
on_day(1, 1).
on_day(2, 2).
after(1, 2).
on_day(3, 2).
on_day(4, 1).
in_timeslot(4, 2).
in_timeslot(4, 3).

% Timeslot Constraints: Event 1 can only be in timeslots 0..3; Event 2 must not overlap with Event 1; Event 3 must be in timeslots 4..7; Event 4 must not be in the same room as Event 1.
valid_timeslot(1, 0..3).
valid_timeslot(2, _).
valid_timeslot(3, 4..7).
valid_timeslot(4, _).
not_same_room(1, 4).

% Generator

% Events: A set of 5 events.
event(1..5).

% Rooms: A set of 3 rooms.
room(1..3).

% Timeslots: A set of 4 timeslots per day over two days.
timeslot(0..7).

% Students: All events are attended by the same set of 4 students.
student(1..4).

% Each event must be scheduled.
1 { scheduled(Event, Room, Timeslot) : room(Room), timeslot(Timeslot) } 1 :- event(Event).

% Only one event can be in a room at a time.
:- scheduled(Event1, Room, Timeslot), scheduled(Event2, Room, Timeslot), Event1 != Event2.

% Events 1 and 2 must be on Day 1.
:- scheduled(Event, _, Timeslot), on_day(Event, 2), timeslot(Timeslot).

% Event 3 must be in timeslots 4..7.
:- scheduled(Event, _, Timeslot), on_day(Event, 1), timeslot(Timeslot), Timeslot < 4.

% Event 4 must not be in the same room as Event 1.
:- scheduled(Event1, Room, _), scheduled(Event2, Room, _), Event1 = 1, Event2 = 4.


% Hard Constraints

% Students must not have overlapping events.
:- scheduled(Student, _, Timeslot1), scheduled(Student, _, Timeslot2), Timeslot1 != Timeslot2.


% The amount of students in a room at the same time should not exceed the capacity of the room
:- scheduled(Student, Room, _), #count{Student2 : scheduled(Student2, Room, _), Student != Student2} > Capacity, room(Room, Capacity).


% Rooms must have the required features for the event.
:- scheduled(Event, Room, _), feature(F), not has_feature(F, Room).


% There must be no more than one event in a room at a time.
:- scheduled(Event1, Room, Timeslot), scheduled(Event2, Room, Timeslot), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- scheduled(Event, _, Timeslot), not valid_timeslot(Event, Timeslot).


% Some events must be scheduled in a specific order.
event(1..6).

1 { scheduled(Event, Room, Timeslot) : room(Room), timeslot(Timeslot) } 1 :- event(Event).

:- scheduled(Event1, Room, Timeslot), scheduled(Event2, Room, Timeslot), Event1 != Event2.

:- scheduled(Event, _, Timeslot), on_day(Event, 2), timeslot(Timeslot).

:- scheduled(Event, _, Timeslot), on_day(Event, 1), timeslot(Timeslot), Timeslot < 4.

:- scheduled(Event1, Room, _), scheduled(Event2, Room, _), Event1 = 1, Event2 = 4.

:- scheduled(Event1, _, Timeslot1), scheduled(Event2, _, Timeslot2), Event1 < Event2, Timeslot1 >= Timeslot2.




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
:- scheduled(Student, _, Timeslot), timeslot(Timeslot), Timeslot = 7.

penalty("LastTimeslot",Student,1) :- scheduled(Student,_,7).


% Students should not have events scheduled in three or more consecutive timeslots
:- student(Student), timeslot(T1), timeslot(T2), timeslot(T3), scheduled(Student,_,T1), scheduled(Student,_,T2), scheduled(Student,_,T3), T3 = T1 + 2.
penalty("ConsecutiveTimeslots", Student, Timeslots) :- student(Student), timeslot(T1), timeslot(T2), timeslot(T3), scheduled(Student, _, T1), scheduled(Student, _, T2), scheduled(Student, _, T3), T3 = T1 + 2, Timeslots = {T1, T2, T3}.


% Students should not have exactly one event in a day
:- student(Student), day(Day), #count{scheduled(Student,_,Timeslot):day(Day),timeslot(Timeslot)} = 1.
penalty("EventIsolation", Student, Day) :- student(Student), day(Day), #count{scheduled(Student, _, Timeslot): day(Day), timeslot(Timeslot)} = 1.

{ penalty("EventIsolation", Student, Day) : day(Day) } 1 :- student(Student).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
