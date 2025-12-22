
% Instance Template
exam(Exam, Duration, Is_large).

student(Student, Exam).

period(Period, Date, Time, Duration, Is_late, Penalty).

room(Room, Capacity, Penalty).

order_constrain(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator

% 1
exam(Exam, _, _) :- exam(Exam).
1 { period(Period, _, _, _, _, _), room(Room, _, _) } 1.

assigned(Exam, Period, Room) :- exam(Exam).
1 { period(Period, _, _, _, _, _), room(Room, _, _) } 1.


% Hard Constraints

% No student sits more than one examination at the same time.
:- student(Student, Exam1), student(Student, Exam2), assigned(Exam1, Period, _), assigned(Exam2, Period, _), Exam1 != Exam2.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- room(Room, Capacity, _), assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), student(Student, Exam1), student(Student, Exam2), #count{Student : student(Student, Exam1), student(Student, Exam2)} > Capacity.


% Period Lengths are not violated.
:- assigned(Exam, Period1, Room), assigned(Exam, Period2, Room), Period1 != Period2, period(_, _, _, Duration1, _, _), period(_, _, _, Duration2, _, _), Duration1 + 1 < Duration2, Period1 > Period2.
:- assigned(Exam, Period1, Room), assigned(Exam, Period2, Room), Period1 != Period2, period(_, _, _, Duration1, _, _), period(_, _, _, Duration2, _, _), Duration1 + 1 < Duration2, Period1 < Period2.


% Some exams must be before other exams.
:- assigned(Exam1, Period1, _), assigned(Exam2, Period2, _), order_constrain(Exam1, Exam2), Period1 > Period2.


% Some exams must be at the same time as other exams.
:- assigned(Exam1, Period1, Room1), assigned(Exam2, Period2, Room2), same_time_constraint(Exam1, Exam2), Period1 != Period2.


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, Period1, Room1), assigned(Exam2, Period2, Room2), different_time_constraint(Exam1, Exam2), Period1 = Period2, Room1 = Room2.


% Some exams must take place in a room with no other exams.
:- assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), own_room_constraint(Exam1), own_room_constraint(Exam2), Exam1 != Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("ConsecutiveExamsInSameDay",consecutive_exams_in_same_day(Student,Day,Period1,Period2),7) :-
student(Student),
period(Period1, Day, _, _, _, _),
period(Period2, Day, _, _, _, _),
Period1 + 1 = Period2,
assigned(Student, Period1, _),
assigned(Student, Period2, _).

penalty("NonConsecutiveExamsInSameDay",non_consecutive_exams_in_same_day(Student,Day,Period1,Period2),5) :-
student(Student),
period(Period1, Day, _, _, _, _),
period(Period2, Day, _, _, _, _),
abs(Period1 - Period2) > 1,
assigned(Student, Period1, _),
assigned(Student, Period2, _).


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("DifferentLengthsInSameRoom",assigned(Exam1,Period,Room),10) :- assigned(Exam1,Period,Room), assigned(Exam2,Period,Room), exam(Exam1,Duration1,_), exam(Exam2,Duration2,_), Duration1 != Duration2.


% Some periods have associated penalties
penalty("PeriodPenalty",assigned(Exam,Period,Room),P) :- period(Period,_,_,_,1,_), assigned(Exam,Period,Room).
penalty("PeriodPenalty",assigned(Exam,Period,Room),P) :- period(Period,_,_,_,P,_), assigned(Exam,Period,Room), P > 1.


% Some rooms have associated penalties
penalty("RoomPenalty",assigned(Exam,Room,Period),(penalty(Room)*1)) :- exam(Exam), room(Room,_,penalty(Room)), assigned(Exam,Room,Period).


% Students should not have more than one exam within a spread of 7 periods.
penalty("ExamSpread", exam_spread(Student, Period1, Period2), 3) :-
    student(Student),
    assigned(Exam1, Period1, Room1),
    assigned(Exam2, Period2, Room2),
    exam(Exam1),
    exam(Exam2),
    Student = Student,
    abs(Period1 - Period2) #< 7.


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExam",assigned(LargeExam,Period,Room),5) :- exam(LargeExam,_,Is_large), Is_large = 1, period(Period,_,_,_,Is_late,_), Is_late = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
