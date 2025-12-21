
% Instance Template
exam(Exam, Duration, Is_large).

student(Student, Exam).

period(Date, Time, Duration, Is_late, Penalty).

room(Room, Capacity, Penalty).

order_constraint(Exam1, Exam2).

same_time(Exam1, Exam2).

different_time(Exam1, Exam2).

own_room(Exam).

% Generator

% 1
assigned(X, Period, Room) :- period(Period), room(Room), exam(X).


% Hard Constraints

% No student sits more than one examination at the same time.
:- assigned(Exam1, _, Room), assigned(Exam2, _, Room), student(Student, Exam1), student(Student, Exam2), Exam1 != Exam2.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
Corrected Code:
:- room(Room, Capacity, _), #count{ S : student(S, X), assigned(K, _, R) : R == Room } > Capacity.


% Period Lengths are not violated.
:- assigned(X, Period1, Room), assigned(Y, Period2, Room), X != Y.
:- assigned(X, _, Room1), assigned(Y, _, Room2), X != Y, Room1 != Room2.

:- assigned(X, _, Day, Period), assigned(Y, _, Day, Period), X != Y.
:- assigned(X, _, Day, Period), assigned(Y, _, Day, Period), room(R), capacity(R) < duration(X), penalty(R) > penalty(X).


:- assigned(X, _, _, R), unavailability_constraint(T, _, R).
:- assigned(X, _, Day, R), order_constraint(X, Y), assigned(Y, Day+1, _).
:- assigned(X, _, Day, R), order_constraint(X, Y), assigned(Y, Day+2, _).


% Some exams must be before other exams.
:- assigned(Exam1, _, Room), assigned(Exam2, _, Room), order_constraint(Exam1, Exam2), assigned(Exam1, Period1), assigned(Exam2, Period2), Period1 < Period2.

:- assigned(Exam1, _, Room), assigned(Exam2, _, Room), order_constraint(Exam2, Exam1), assigned(Exam1, Period1), assigned(Exam2, Period2), Period1 > Period2.


% Some exams must be at the same time as other exams.
:- assigned(Exam1, _, Room), assigned(Exam2, _, Room), same_time(Exam1, Exam2), Exam1 != Exam2.
:- assigned(Exam1, _, Room), assigned(Exam2, _, Room), different_time(Exam1, Exam2), same_time(Exam1, Exam2).


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, _, Period), assigned(Exam2, _, Period), Exam1 != Exam2.


% Some exams must take place in a room with no other exams.
:- assigned(X, _, R), own_room(X), assigned(Y, _, R), Y != X.




% Soft Constraints

% Students should not have more than one exam in the same day
has_exam(Student, Day) :- student(Student, Exam), period(Day, _, _, _, _), exam(Exam, _, _).
penalty("SameDayExams", same_day_exams(Student,Day),7) :- has_exam(Student,Day), has_exam(Student+1,Day), period(Day-1,_,_,_,_).
penalty("SameDayExams", same_day_exams(Student,Day),5) :- has_exam(Student,Day), has_exam(Student+1,Day), not period(Day-1,_,_,_,_).


% Exams of different lengths should not be scheduled in the same room in the same period
:- assigned(X1, P, R), assigned(X2, P, R), exam(X1, D1), exam(X2, D2), D1 != D2.
penalty("LengthMismatch",1) :- assigned(X1,P,R), assigned(X2,P,R), exam(X1,D1), exam(X2,D2), D1 != D2.


% Some periods have associated penalties
penalty("PeriodPenalty", scheduled(Exam, Period, Room), P) :- period(Period, _, _, P), assigned(Exam, Period, Room).


% Some rooms have associated penalties
penalty("RoomPenalty", scheduled(X, R), P) :- assigned(X, _, R), room(R, _, P).


% Students should not have more than one exam within a spread of 7 periods.
Corrected Code:
penalty("ExamSpread", HasExams, 3 * (HasExams - 1)) :- student(Student), HasExams = #count{E1, E2 : exam(E1), exam(E2), P : assigned(E1, P, _), assigned(E2, P, _), P1, P2 : period(P1), period(P2), P1 - P2 = 7}.


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExams", scheduled(X, Period, Room), 5) :- exam(X), large(X), Is_late = period(Period, _, _, Is_late, P), P = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
