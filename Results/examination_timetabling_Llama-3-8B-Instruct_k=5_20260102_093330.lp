
% Instance Template
exam(Exam, Duration, Is_large).

student(Student, Exam).

period(Date, Time, Duration, Is_late, Penalty).

room(Room, Capacity, Penalty).

order_constraint(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator

% 1
1 { assigned(Exam, Period, Room) : period(Period), room(Room) } 1 :- exam(Exam,_,_).


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
:- same_time_constraint(Exam1, Exam2), assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), Exam1 != Exam2.


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, Period1, Room), assigned(Exam2, Period2, Room), Exam1 != Exam2, Period1 == Period2.


% Some exams must take place in a room with no other exams.
:- assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), Exam1 != Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("SameDayExams",same_day_exams(Student,Day),7) :- student(Student,Exam), assigned(Exam,Period,Room), period(Date,_,_,_,_) = { period(_,Time,_,_,_) : assigned(Exam,_,Room), student(Student,Exam), Date = Day }, Period = { period(_,Time,_,_,_) : assigned(Exam,_,Room), student(Student,Exam), Date = Day }, Period = Period-1, Period = { period(_,Time,_,_,_) : assigned(Exam,_,Room), student(Student,Exam), Date = Day }.
penalty("SameDayExams",same_day_exams(Student,Day),5) :- student(Student,Exam), assigned(Exam,Period,Room), period(Date,_,_,_,_) = { period(_,Time,_,_,_) : assigned(Exam,_,Room), student(Student,Exam), Date = Day }, Period = { period(_,Time,_,_,_) : assigned(Exam,_,Room), student(Student,Exam), Date = Day }, Period = { period(_,Time,_,_,_) : assigned(Exam,_,Room), student(Student,Exam), Date = Day }, Period = Period-1, not Period = { period(_,Time,_,_,_) : assigned(Exam,_,Room), student(Student,Exam), Date = Day }-1.


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("ExamLength",exam_length_mismatch(Room,Period),10) :- assigned(Exam,Period,Room), exam(Exam,Duration1), exam(Exam2,Duration2), Duration1 != Duration2, Exam != Exam2.


% Some periods have associated penalties
penalty("PeriodPenalty",assigned(Exam,Period,Room),(P*1)) :- exam(Exam,_,_), period(Period,_,_,_,P), assigned(Exam,Period,Room).


% Some rooms have associated penalties
penalty("RoomPenalty",assigned(Exam,Period,Room),(P*1)) :- exam(Exam,_,_), period(_,_,_,_,_), room(Room,P), assigned(Exam,Period,Room).


% Students should not have more than one exam within a spread of 7 periods.
Corrected Code:
penalty("ExamSpread",has_two_exams(Student,Exam1,Exam2),3) :- 
    student(Student,Exam1),
    student(Student,Exam2),
    exam(Exam1,_,_),
    exam(Exam2,_,_),
    period(Period1,_,_,_,_),
    period(Period2,_,_,_,_),
    P1 = #min{P : period(P,_,_,_,_) | [P1]},
    P2 = #min{P : period(P,_,_,_,_) | [P2]},
    abs(P1 - P2) <= 7,
    P1 < P2.


% An exam that is large should not be scheduled in a period that is late.
penalty("LargeExamsLate",exam(Exam,_,Is_large),(Is_large*1)*5) :- assigned(Exam,Period,Room), period(Date,Time,_,Is_late, Penalty), Is_large = 1, Is_late = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
