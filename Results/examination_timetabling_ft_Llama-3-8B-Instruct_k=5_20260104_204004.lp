
exam(Exam, Duration, Is_large).

student(Student, Exam).

period(Date, Time, Duration, Is_late, Penalty).

room(Room, Capacity, Penalty).

order_constraint(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator

1 { assigned(Exam, Period, Room) : period(Period), room(Room) } 1 :- exam(Exam).


% Hard Constraints

% No student sits more than one examination at the same time.
:- assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), student(Student, Exam1), student(Student, Exam2).


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- room(Room, Capacity, _), #count{Student : student(Student, Exam), assigned(Exam, _, Room)} > Capacity.


% Period Lengths are not violated.
:- assigned(Exam, Period1, Room), assigned(Exam, Period2, Room), period(Period1, _, Duration, _, _), period(Period2, _, Duration, _, _), Period1 != Period2.


% Some exams must be before other exams.
:- assigned(Exam1, Period1, Room), assigned(Exam2, Period2, Room), order_constraint(Exam1, Exam2), Period1 > Period2.


% Some exams must be at the same time as other exams.
:- same_time_constraint(Exam1, Exam2), assigned(Exam1, Period, Room), assigned(Exam2, Period, Room).


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, Period1, Room), assigned(Exam2, Period2, Room), Exam1 != Exam2, Period1 == Period2.


% Some exams must take place in a room with no other exams.
:- assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), Exam1 != Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("SameDayExams",same_day_exams(Exam1,Exam2),7) :- exam(Exam1), exam(Exam2), student(Student,Exam1), student(Student,Exam2), same_period(Exam1,Exam2).
penalty("SameDayExams",same_day_exams(Exam1,Exam2),5) :- exam(Exam1), exam(Exam2), student(Student,Exam1), student(Student,Exam2), same_day(Exam1,Exam2), not same_period(Exam1,Exam2).
same_day(Exam1,Exam2) :- exam(Exam1), exam(Exam2), student(Student,Exam1), student(Student,Exam2), period(Period1,_,_,_,_), period(Period2,_,_,_,_), same_day(Period1,Period2).
same_day(Period1,Period2) :- Period1 = Period2, Period1 = { period(Period,_,_,_,_) }.
same_day(Period1,Period2) :- Period1 = Period2-1, period(Period1,_,_,_,_).
same_day(Period1,Period2) :- Period1 = Period2+1, period(Period1,_,_,_,_).
same_period(Exam1,Exam2) :- exam(Exam1), exam(Exam2), period(Period1,_,_,_,_), period(Period2,_,_,_,_), Period1 = Period2, period(Period1,_,_,_,_).


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("ExamLength",exam_length_diff(Exam1,Exam2),10) :- assigned(Exam1,Period,Room), assigned(Exam2,Period,Room), exam(Exam1,_,L1), exam(Exam2,_,L2), L1 != L2.


% Some periods have associated penalties
penalty("PeriodPenalty", assigned(Exam, Period, Room), Penalty) :- assigned(Exam, Period, Room), period(Period, _, _, _, Penalty).


% Some rooms have associated penalties
penalty("RoomPenalty", assigned(Exam, Period, Room), Penalty) :- exam(Exam), period(Period, _, _, _, _), room(Room, _, Penalty), assigned(Exam, Period, Room).


% Students should not have more than one exam within a spread of 7 periods.
Corrected Code:
penalty("ExamSpread",exam_spread(Student,Exam1,Exam2),3) :- student(Student,Exam1), student(Student,Exam2), exam(Exam1,_,_), exam(Exam2,_,_), period(Period1,_,_,_,_), period(Period2,_,_,_,_), Period2 = Period1 + 7, Period1 = Period2 - 7.


% An exam that is large should not be scheduled in a period that is late.
penalty("LargeExamLate", Exam, 5) :- exam(Exam, _, Is_large), Is_large, period(_, _, _, Is_late, _), Is_late.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
