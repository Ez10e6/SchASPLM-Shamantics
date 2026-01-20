
exam(Exam, Duration, Is_Large).

student(Student, Exam).

period(P, Date, Time, Duration, Is_Late, Penalty).

room(Room, Capacity, Penalty).

order_constrain(E1, E2).

same_time_constraint(E1, E2).

different_time_constraint(E1, E2).

own_room_constraint(Exam).

% Generator

N_exams { assigned(Exam, Room, Period) : exam(Exam, _, _), room(Room, _, _), period(Period, _, _, _, _, _) }.


% Hard Constraints

% No student sits more than one examination at the same time.
:- student(Student, Exam1), student(Student, Exam2), Exam1 != Exam2, assigned(Exam1, _, Period), assigned(Exam2, _, Period).


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- assigned(Exam1, Room, Period), assigned(Exam2, Room, Period), Exam1 != Exam2, student(Student, Exam1), student(Student, Exam2), #count{Exam : assigned(Exam, Room, Period), student(Student, Exam)} > Capacity, room(Room, Capacity, _).


% Period Lengths are not violated.
:- assigned(Exam, Room, Period), period(_, _, _, Duration, _, _), period(_, _, _, Duration2, _, _), Duration != Duration2.


% Some exams must be before other exams.
:- order_constrain(E1, E2), not assigned(E1, _, P1), not assigned(E2, _, P2), P1 < P2.


% Some exams must be at the same time as other exams.
:- assigned(E1, _, P), assigned(E2, _, P), same_time_constraint(E1, E2).


% Some exams must be at a differen time than other exams.
:- assigned(E1, _, P1), assigned(E2, _, P2), different_time_constraint(E1, E2).


% Some exams must take place in a room with no other exams.
:- assigned(E1, R, P), assigned(E2, R, P), E1 != E2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("ExamOverlap",student(Student,E1),7) :- student(Student,E1), student(Student,E2), E1 < E2, assigned(E1,_,P1), assigned(E2,_,P2), P1 < P2, P1 + 1 = P2, period(P1,_,_,_,_,_), period(P2,_,_,_,_,_).
penalty("ExamOverlap",student(Student,E1),5) :- student(Student,E1), student(Student,E2), E1 < E2, assigned(E1,_,P1), assigned(E2,_,P2), P1 < P2, P1 + 1 != P2, period(P1,_,_,_,_,_), period(P2,_,_,_,_,_).


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("DifferentLength",assigned(E1, Room, Period),10) :- assigned(E1, Room, Period), assigned(E2, Room, Period), exam(E1, Dur1, _), exam(E2, Dur2, _), Dur1 != Dur2.


% Some periods have associated penalties
penalty("PeriodPenalty",exam(Exam,_,_),P*Penalty) :- assigned(Exam,Room,Period), period(Period,_,_,_,Penalty,_).


% Some rooms have associated penalties
penalty("RoomPenalty",Room,Penalty*(N-1)) :- assigned(Exam,Room,Period), room(Room,_,Penalty), N = #count{P : assigned(Exam,Room,P)}.


% Students should not have more than one exam within a spread of 7 periods.
penalty("ExamSpread",student(S),3) :- student(S), N = #count{P1,P2 : assigned(E1,S,P1), assigned(E2,S,P2), P1 < P2, P2 - P1 <= 7}, N > 1.


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExam", Exam, 5) :- assigned(Exam, Room, Period), period(Period, _, _, _, Is_Late, _), exam(Exam, _, Is_Large), Is_Large = "large", Is_Late = "late".




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
