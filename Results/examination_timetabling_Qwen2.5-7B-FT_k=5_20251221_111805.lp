
% Exam Variables
exam(E, Duration, Large).

% Student Variables
student(S, Exam).

% Period Variables
period(P, Date, Time, Duration, Late, Penalty).

% Room Variables
room(R, Capacity, Penalty).

% Order Constraints
order(E1, E2).

% Same Time Constraints
same_time(E1, E2).

% Different Time Constraints
diff_time(E1, E2).

% Own Room Constraints
own_room(E).

% Generator

% Exam Variables
exam(E, Duration, Large).

% Student Variables
student(S, Exam).

% Period Variables
period(P, Date, Time, Duration, Late, Penalty).

% Room Variables
room(R, Capacity, Penalty).

% Select one period and one room for each exam
1 { assigned(E, R, P) : room(R), period(P) } 1 :- exam(E).

% Room capacity must be at least as large as the number of students taking the exam
:- assigned(E, R, _), exam(E, _, Large), student(S, E), room(R, Capacity, _), Capacity < Large.

% Late exams must be scheduled after their deadline
:- assigned(E, _, P), exam(E, _, _), late(E), period(P, _, Time, Duration, Late, _), period(P1, _, Time1, Duration1, _, _), P < P1.

% Exams with order constraint must be scheduled consecutively
:- assigned(E1, _, P), assigned(E2, _, P), order(E1, E2), period(P1, _, Time1, Duration1, _, _), period(P2, _, Time2, Duration2, _, _), P1 = P2, Time1 + Duration1 > Time2.

% Exams with same time constraint must be scheduled in the same room
:- assigned(E1, R1, P), assigned(E2, R2, P), same_time(E1, E2), R1 != R2.

% Exams with different time constraint must not be scheduled in the same room
:- assigned(E1, R, P), assigned(E2, R, P), diff_time(E1, E2), E1 != E2.

% Only one exam can be in a room at a time
:- assigned(E1, R, P), assigned(E2, R, P), E1 != E2.

% Each room can be used at most once per day
{ assigned(E, R, P) : exam(E), period(P) } 1 :- room(R), day(D), period(P), D = day(P).

% Each room can be used at most twice per day
{ assigned(E, R, P) : exam(E), period(P) } 2 :- room(R), day(D), period(P), D = day(P).

% At least one exam must be scheduled each day
:- day(D), not assigned(_, _, P), D = day(P).


% Hard Constraints

% No student sits more than one examination at the same time.
:- student(S, E1), student(S, E2), assigned(E1, _, P), assigned(E2, _, P), E1 != E2.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- assigned(E1, R, P), assigned(E2, R, P), student(S1, E1), student(S2, E2), S1 != S2, room(R, Capacity, _), Capacity < 2.


% Period Lengths are not violated.
:- assigned(E, _, P), period(P, _, Time, Duration, _, _), Time + Duration > 18.


% Some exams must be before other exams.
:- assigned(E1, _, P), assigned(E2, _, P), order(E1, E2), period(P1, _, Time1, Duration1, _, _), period(P2, _, Time2, Duration2, _, _), P1 = P2, Time1 + Duration1 >= Time2.


% Some exams must be at the same time as other exams.
:- assigned(E1, R1, P), assigned(E2, R2, P), same_time(E1, E2), R1 != R2.


% Some exams must be at a differen time than other exams.
:- assigned(E1, _, P), assigned(E2, _, P), diff_time(E1, E2), E1 != E2.


% Some exams must take place in a room with no other exams.
:- assigned(E1, R, P), assigned(E2, R, P), E1 != E2.




% Soft Constraints

% Students should not have more than one exam in the same day
:- student(S, E1), student(S, E2), assigned(E1, _, P), assigned(E2, _, P), E1 != E2.

:- student(S, E1), student(S, E2), assigned(E1, _, P), assigned(E2, _, P), E1 != E2, period(P, D, T1, Dur1, _, _), period(P1, D, T2, Dur2, _, _), P = P1, T1 + Dur1 = T2.

:- student(S, E1), student(S, E2), assigned(E1, _, P), assigned(E2, _, P), E1 != E2, period(P, D, T1, Dur1, _, _), period(P1, D, T2, Dur2, _, _), P = P1, T1 + Dur1 != T2.


% Exams of different lengths should not be scheduled in the same room in the same period
:- assigned(E1, R, P), assigned(E2, R, P), exam(E1, Dur1, _), exam(E2, Dur2, _), Dur1 != Dur2.
penalty("DifferentLengths",10) :- assigned(E1, R, P), assigned(E2, R, P), exam(E1, Dur1, _), exam(E2, Dur2, _), Dur1 != Dur2.


% Some periods have associated penalties
has_penalty(P, W) :- period(P, _, _, _, _, W).
scheduled_with_penalty(E, P) :- assigned(E, _, P), has_penalty(P, _).
penalty(P, N) :- N = #count{E : scheduled_with_penalty(E, P), has_penalty(P, W)}, Penalty = weight(E, P).


% Some rooms have associated penalties
room(R, Capacity, Penalty) :- is_penalty(R, Penalty).

:- assigned(E, R, _), room(R, _, Penalty), is_penalty(R, Penalty).

total_penalty(P) :- P = #sum{R : assigned(E, R, _), room(R, _, Penalty), is_penalty(R, Penalty)}.


#minimize { P : total_penalty(P) }.


% Students should not have more than one exam within a spread of 7 periods.
spread(P1, P2) :- student(S, E1), student(S, E2), assigned(E1, _, P1), assigned(E2, _, P2), P2 - P1 <= 7.
penalty("ExamSpread",spread(P1, P2),3) :- spread(P1, P2).


% An exam that is large should not be scheduled in a period that is late.
:- assigned(E, _, P), exam(E, _, Large), late(P), Large.
Penalty = #count{E : assigned(E, _, P), exam(E, _, Large), late(P) } * 5.


penalty("LateLargeExam", assigned(E, _, P), 5) :- assigned(E, _, P), exam(E, _, Large), late(P), Large.
Penalty = #count{E : penalty("LateLargeExam", assigned(E, _, P), 5) } * 5.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
