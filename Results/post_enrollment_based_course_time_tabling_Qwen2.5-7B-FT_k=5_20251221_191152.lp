
% Instance Template
event(Event).

room(Room, Capacity).

timeslot(Day, Hour).

student(Student).

feature(F, Event).

room_feature(F, Room).

precedence_constraint(Event1, Event2).

timeslot_constraint(Event, Timeslot).

% Generator

% Assignment of events to timeslots and rooms
1 { assigned(Event, Room, Timeslot) : room(Room, _), timeslot(Timeslot) } 1 :- event(Event).


% Hard Constraints

% Students must not have overlapping events.
:- assigned(Event1, _, Timeslot), assigned(Event2, _, Timeslot), student(Student), Event1 != Event2, overlaps(Event1, Student), overlaps(Event2, Student).


% The amount of students in a room at the same time should not exceed the capacity of the room
:- room(Room, Capacity),
#count{Timeslot : assigned(Event1, Room, Timeslot), feature("student", Event1)} > Capacity,
#count{Timeslot : assigned(Event2, Room, Timeslot), feature("student", Event2)} > Capacity.


% Rooms must have the required features for the event.
:- assigned(Event, Room, _), room_feature(F, Room), not feature(F, Event).


% There must be no more than one event in a room at a time.
:- assigned(Event1, Room, Timeslot), assigned(Event2, Room, Timeslot), Event1 != Event2.


% Events may only be scheduled in designated timeslots.
:- assigned(Event, Room, Timeslot), not timeslot_constraint(Event, Timeslot).


% Some events must be scheduled in a specific order.
:- assigned(Event1, _, _), assigned(Event2, _, _), precedence_constraint(Event1, Event2), not assigned(Event2, _, _).




% Soft Constraints

% Students should not have events scheduled in the last timeslot of the day
penalty("LastTimeslot",student(S),1) :- assigned(Event,S,_,LastTimeslot), timeslot(_,LastTimeslot), LastTimeslot = #max{Hour : timeslot(_,Hour)}.


% Students should not have events scheduled in three or more consecutive timeslots
penalty("ConsecutiveTimeslots", student(S), N) :-
    3 #<= N,
    { timeslot(T) : assigned(S, _, T) } #= N,
    N_consecutive(0, N, Ts),
    #count{ T : member(T, Ts) } #> 2.


N_consecutive(N, [T|TsRest], [T|Ts]) :- Ts = [_|_], N1 is N + 1, N_consecutive(N1, TsRest, Ts).
penalty("ConsecutiveTimeslots", student(S), N) :-
    3 #<= N,
    { timeslot(T) : assigned(S, _, T) } #= N,
    N_consecutive(0, Ts, _),
    #count{ T : member(T, Ts) } #> 2.

N_consecutive(N, [], []).
N_consecutive(N, [T|TsRest], [T|Ts]) :- Ts = [_|_], N1 is N + 1, N_consecutive(N1, TsRest, Ts).
penalty("ConsecutiveTimeslots", student(S), N) :-
    3 #<= N,
    { timeslot(T) : assigned(S, _, T) } #= N,
    N_consecutive(0, Ts, _),
    #count{ T : member(T, Ts) } #> 2.


% Students should not have exactly one event in a day
penalty("SingleEventPerDay", #count{Timeslot : assigned(Event1, _, Timeslot), assigned(Event2, _, Timeslot), Event1 != Event2, student(Student), event(Event1), event(Event2)} - #count{Timeslot : assigned(Event, _, Timeslot), student(Student), event(Event)}, 1).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
