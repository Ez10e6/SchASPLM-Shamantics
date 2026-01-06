
% Instance Template
exam(Exam, Duration, Is_large).

student(Student, Exam).

period(P, Date, Time, Duration, Is_late, Penalty).

room(R, Capacity, Penalty).

order_constrain(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator

% 1
{ exam(Exam, _, _) : student(Student, Exam) } 1 :- exam(Exam, _, _), period(P, _, _, _, _, _), room(R, _, _).

{ assigned(Exam, P, R) : period(P, _, _, _, _, _), room(R, _, _) } 1 :- exam(Exam, _, _).

assigned(Exam, P, R) :- exam(Exam, _, _), period(P, _, _, _, _, _), room(R, _, _).


% Hard Constraints

% No student sits more than one examination at the same time.
{ assigned(Exam, P, R) : period(P, _, _, _, _, _), room(R, _, _) } 1 :- exam(Exam, _, _), student(Student, Exam), { assigned(Student, P, R) : period(P, _, _, _, _), room(R, _, _) } 1.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- room(R, Capacity, _), assigned(Exam1, P1, R), assigned(Exam2, P2, R), period(P1, _, _, _, _, _), period(P2, _, _, _, _, _), P1 != P2, #count{Student: student(Student, Exam1), student(Student, Exam2)} >= Capacity.


% Period Lengths are not violated.
{ assigned(Exam, P, R) : period(P, _, _, Dur1, _, _), room(R, _, Dur2) } Dur1 + Dur2 - 1.


% Some exams must be before other exams.
assigned(Exam1, P1, _) :- assigned(Exam2, P2, _), order_constrain(Exam1, Exam2), P1 < P2.


% Some exams must be at the same time as other exams.
same_time_constraint(Exam1, Exam2) :- order_constrain(Exam1, Exam2).


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, P1, R1), assigned(Exam2, P2, R2), different_time_constraint(Exam1, Exam2), P1 = P2, R1 = R2.


% Some exams must take place in a room with no other exams.
{ assigned(Exam, P, R) : period(P, _, _, _, _, _), room(R, _, _) } 1 :- exam(Exam, _, _), own_room_constraint(Exam).




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("ConsecutiveExamsSameDay",assigned(Exam1,P1,R1),7) :-
assigned(Exam1,P1,R1),
assigned(Exam2,P2,R2),
P1 = P2,
day(P1,Day1),
day(P2,Day1),
period(P1,_,_,_,_,_),
period(P2,_,_,_,_,_),
abs(period_number(P1)-period_number(P2)) = 1.

penalty("NonConsecutiveExamsSameDay",assigned(Exam1,P1,R1),5) :-
assigned(Exam1,P1,R1),
assigned(Exam2,P2,R2),
P1 = P2,
day(P1,Day1),
day(P2,Day1),
period(P1,_,_,_,_,_),
period(P2,_,_,_,_,_),
abs(period_number(P1)-period_number(P2)) > 1.


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("SameRoomDifferentLength",assigned(Exam1, P, R),10) :- assigned(Exam1, P, R), assigned(Exam2, P, R), exam(Exam1, L1, _), exam(Exam2, L2, _), L1 != L2.


% Some periods have associated penalties
penalty("PeriodPenalty",assigned(Exam,P,R),P_penalty) :- exam(Exam,_,_), period(P,_,_,_,_,P_penalty), assigned(Exam,P,R).


% Some rooms have associated penalties
penalty("RoomPenalty",assigned(Exam,P,R),Penalty) :- exam(Exam,_,_), period(P,_,_,_,_,_), room(R,_,Penalty), has_penalty(R).


% Students should not have more than one exam within a spread of 7 periods.
penalty("ExamSpread",student(Student,Exam1),3) :-
student(Student,Exam1),
student(Student,Exam2),
assigned(Exam1,P1,R1),
assigned(Exam2,P2,R2),
P1 < P2,
P2 - P1 <= 7,
P2 - P1 > 1.


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExam",assigned(LargeExam,P,Room),5) :- exam(LargeExam,_,Is_large), Is_large = 1, period(P,_,_,_,Is_late,_), Is_late = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
