
exam(Exam, Duration, Is_Large).

student(Student, Exam).

period(Date, Time, Duration, Is_Late, Penalty).

room(Room, Capacity, Penalty).

order_constraint(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator

1 { assigned(Exam, Room, Period) : room(Room, _, _), period(Period, _, _, _, _) } 1 :- exam(Exam).


% Hard Constraints

% No student sits more than one examination at the same time.
:- student(Student, Exam1), student(Student, Exam2), Exam1 != Exam2, assigned(Exam1, _, Period), assigned(Exam2, _, Period).


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- assigned(Exam1, Room, Period), assigned(Exam2, Room, Period), Exam1 != Exam2, student(Student, Exam1), student(Student, Exam2), room(Room, Capacity, _), #count{Student : student(Student, Exam1), student(Student, Exam2)} > Capacity.


% Period Lengths are not violated.
:- assigned(Exam1, _, Period1), assigned(Exam2, _, Period2), Period1 != Period2, duration(Exam1, Duration1), duration(Exam2, Duration2), Duration1 + Duration2 > Period1.


% Some exams must be before other exams.
:- assigned(Exam1, _, Period1), assigned(Exam2, _, Period2), order_constraint(Exam1, Exam2), Period1 >= Period2.


% Some exams must be at the same time as other exams.
:- assigned(Exam1, _, Period), assigned(Exam2, _, Period), same_time_constraint(Exam1, Exam2).


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, _, Period1), assigned(Exam2, _, Period2), different_time_constraint(Exam1, Exam2), Period1 == Period2.


% Some exams must take place in a room with no other exams.
:- assigned(Exam1, Room, Period), assigned(Exam2, Room, Period), Exam1 != Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("ExamOverlap",student(Student,Exam1),7) :- student(Student,Exam1), student(Student,Exam2), assigned(Exam1,_,Period1), assigned(Exam2,_,Period2), Period1 != Period2, Period1 < Period2, Period1 + 1 == Period2.
penalty("ExamOverlap",student(Student,Exam1),5) :- student(Student,Exam1), student(Student,Exam2), assigned(Exam1,_,Period1), assigned(Exam2,_,Period2), Period1 != Period2, Period1 < Period2, Period1 + 1 != Period2.


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("DifferentLengths",assigned(Exam1,Room,Period),10) :- assigned(Exam1,Room,Period), assigned(Exam2,Room,Period), exam(Exam1,Duration1), exam(Exam2,Duration2), Duration1 != Duration2.


% Some periods have associated penalties
penalty("PeriodPenalty",assigned(Exam,Room,Period),P*1) :- period(Period,_,_,P,_).


% Some rooms have associated penalties
penalty("RoomPenalty",assigned(Exam,Room,Period),Penalty) :- assigned(Exam,Room,Period), room(Room,Penalty,_).


% Students should not have more than one exam within a spread of 7 periods.
penalty("ExamSpread",student(Student),3) :- student(Student,Exam1), student(Student,Exam2), assigned(Exam1,_,Period1), assigned(Exam2,_,Period2), Period1 < Period2, Period2 - Period1 <= 7.


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExam",exam(Exam),5) :- exam(Exam), Is_Large = 1, period(Period, _, _, Is_Late, _), assigned(Exam, _, Period), Is_Late = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
