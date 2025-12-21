
% Instance Template
exam(Exam, Duration, is_large).

student(Student, Exam).

period(Date, Time, Duration, is_late, penalty).

room(Room, Capacity, penalty).

order_constraint(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator

% Assignment of exams to periods and rooms
1 { assigned(Exam, Room, Period) : room(Room, _), period(Period) } 1 :- exam(Exam).


% Hard Constraints

% No student sits more than one examination at the same time.
:- student(Student, Exam1), student(Student, Exam2), assigned(Exam1, _, Period), assigned(Exam2, _, Period), Exam1 != Exam2.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- room(Room, Capacity, _), assigned(Exam1, Room, Period), assigned(Exam2, Room, Period), student(Student, Exam1), student(Student, Exam2), Exam1 != Exam2,
#count{Exam : assigned(Exam, Room, Period)} > Capacity.


% Period Lengths are not violated.
:- assigned(_, _, Period1, _), assigned(_, _, Period2, _), Period1 != Period2, abs(Period1 - Period2) < 1.


% Some exams must be before other exams.
:- assigned(Exam1, _, Period1), assigned(Exam2, _, Period2), order_constraint(Exam1, Exam2), Period1 >= Period2.


% Some exams must be at the same time as other exams.
:- assigned(Exam1, _, Period), assigned(Exam2, _, Period), same_time_constraint(Exam1, Exam2), Exam1 != Exam2.


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, _, Period1), assigned(Exam2, _, Period2), different_time_constraint(Exam1, Exam2), Period1 = Period2.


% Some exams must take place in a room with no other exams.
:- assigned(Exam1, Room, Period), assigned(Exam2, Room, Period), own_room_constraint(Exam1), own_room_constraint(Exam2), Exam1 != Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("ConsecutiveExams",assigned(Exam1,Room1,Day,Period1),7) :-
assigned(Exam1,Room1,Day,Period1),
assigned(Exam2,Room2,Day,Period2),
Period1 < Period2,
Period2 - Period1 = 1,
Exam1 != Exam2.

penalty("NonConsecutiveExams",assigned(Exam1,Room1,Day,Period1),5) :-
assigned(Exam1,Room1,Day,Period1),
assigned(Exam2,Room2,Day,Period2),
Period1 < Period2,
Period2 - Period1 > 1,
Exam1 != Exam2.


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("DifferentLengthsInSameRoom",assigned(Exam1,Room,Period),10) :-
assigned(Exam1,Room,Period),
assigned(Exam2,Room,Period),
exam(Exam1,Duration1,_),
exam(Exam2,Duration2,_),
Duration1 != Duration2.


% Some periods have associated penalties
penalty("PeriodPenalty",assigned(Exam,Room,Period),P) :- assigned(Exam,Room,Period), period(Period,P).


% Some rooms have associated penalties
penalty("RoomPenalty", Exam, P) :- assigned(Exam, Room, _), room(Room, P).


% Students should not have more than one exam within a spread of 7 periods.
penalty("ExamSpread",student(Student),3) :-
assigned(Exam1, _, Period1),
assigned(Exam2, _, Period2),
student(Student, Exam1),
student(Student, Exam2),
Period1 < Period2,
Period2 - Period1 <= 7,
#count{P : assigned(Student, _, P), P >= Period1, P < Period2} = 2.


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExam",assigned(Exam,Room,Period),(is_late(Period)*5)) :- exam(Exam,is_large), assigned(Exam,Room,Period), period(Period,is_late).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
