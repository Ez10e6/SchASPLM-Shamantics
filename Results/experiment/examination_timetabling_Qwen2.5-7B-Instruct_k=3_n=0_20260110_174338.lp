

exam(Exam, Duration, Is_large).

student(Student, Exam).

period(P, Date, Time, Duration, Is_late, Penalty).

room(R, Capacity, Penalty).

order_constrain(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator


{ exam_to_period(Exam, P, Date, Time, Duration, Is_late, Penalty, Room) : room(Room, _, _), period(P, Date, Time, Duration, Is_late, Penalty) } 1 :- exam(Exam).


% Hard Constraints

% No student sits more than one examination at the same time.
:- student(Student, Exam1), student(Student, Exam2), exam_to_period(Exam1, _, _, _, _, _, _, _), exam_to_period(Exam2, _, _, _, _, _, _, _), different_time_constraint(Exam1, Exam2).


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- room(R, Capacity, _), exam_to_period(Exam1, R, _, _, _, _, _), exam_to_period(Exam2, R, _, _, _, _, _), student(Student, Exam1), student(Student, Exam2), Exam1 != Exam2, #count{Exam : exam_to_period(Exam, R, _, _, _, _, _, _), student(Student, Exam)} > Capacity.


% Period Lengths are not violated.
:- exam_to_period(Exam, P1, Date1, Time1, Duration1, Is_late1, Penalty1, Room1), exam_to_period(Exam, P2, Date2, Time2, Duration2, Is_late2, Penalty2, Room2), P1 != P2, Date1 = Date2, abs(Time1 - Time2) < Duration1 + Duration2.


% Some exams must be before other exams.
:- exam_to_period(Exam1, _, _, _, _, _, _, _), exam_to_period(Exam2, _, _, _, _, _, _, _), order_constrain(Exam1, Exam2), exam_to_period(Exam2, P2, Date2, Time2, _, _, _, _), exam_to_period(Exam1, P1, Date1, Time1, _, _, _, _), P1 >= P2, Date1 >= Date2, Time1 >= Time2.


% Some exams must be at the same time as other exams.
same_time_constraint(Exam1, Exam2).


% Some exams must be at a differen time than other exams.
:- exam_to_period(Exam1, P1, Date1, Time1, _, _, _, _), exam_to_period(Exam2, P2, Date2, Time2, _, _, _, _), different_time_constraint(Exam1, Exam2), P1 == P2, Date1 == Date2, Time1 != Time2.


% Some exams must take place in a room with no other exams.
own_room_constraint(Exam).




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("ExamOverlapDayConsecutive", exam_to_period(Exam1, P1, Date1, Time1, Duration1, Is_late1, _, Room1), 7) :-
    exam_to_period(Exam2, P2, Date2, Time2, Duration2, Is_late2, _, Room2),
    Date1 = Date2,
    abs(P2 - P1) =< 1,
    exam(Exam1),
    exam(Exam2),
    not Exam1 = Exam2.

penalty("ExamOverlapDayNonConsecutive", exam_to_period(Exam1, P1, Date1, Time1, Duration1, Is_late1, _, Room1), 5) :-
    exam_to_period(Exam2, P2, Date2, Time2, Duration2, Is_late2, _, Room2),
    Date1 = Date2,
    abs(P2 - P1) > 1,
    exam(Exam1),
    exam(Exam2),
    not Exam1 = Exam2.
penalty("ExamOverlapDayNotConsecutive",exam_to_period(Exam1,P1,Date1,Time1,Duration1,Is_late1,_,Room1),5) :- exam_to_period(Exam1,P1,Date1,Time1,Duration1,Is_late1,_,Room1), exam_to_period(Exam2,P2,Date2,Time2,Duration2,Is_late2,_,Room2), Date1 = Date2, abs(P2-P1) > 1, exam(Exam1), exam(Exam2), not Exam1 = Exam2.


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("SameRoomDifferentLength", exam_to_period(Exam1, P, Date, Time, Duration1, Is_late1, Penalty1, Room), 10) :-
    exam_to_period(Exam1, P, Date, Time, Duration1, Is_late1, Penalty1, Room),
    exam_to_period(Exam2, P, Date, Time, Duration2, Is_late2, Penalty2, Room),
    exam(Exam1, Duration1, _),
    exam(Exam2, Duration2, _),
    Duration1 \= Duration2.


% Some periods have associated penalties
penalty("PeriodPenalty",exam_to_period(Exam,P,Date,Time,Duration,Is_late,Penalty,Room),Penalty*(1)) :- exam_to_period(Exam,P,Date,Time,Duration,Is_late,Penalty,Room), period(P,Date,Time,Duration,Is_late,Penalty).


% Some rooms have associated penalties
penalty("RoomPenalty",exam_to_period(Exam,P,Date,Time,Duration,Is_late,Penalty,Room),Penalty*room(Room,_,Penalty)) :- exam_to_period(Exam,P,Date,Time,Duration,Is_late,Penalty,Room), room(Room,_,Penalty).


% Students should not have more than one exam within a spread of 7 periods.
penalty("ExamSpread",student(Student,Exam1),3) :-
student(Student,Exam1),
student(Student,Exam2),
exam_to_period(Exam1,P1,Date1,Time1,Duration1,IsLate1,_,Room1),
exam_to_period(Exam2,P2,Date2,Time2,Duration2,IsLate2,_,Room2),
abs(P1-P2) <= 7,
P1 < P2.


% An exam that is large should not be scheduled in a period that is late.
penalty("LatePeriodForLargeExam",exam_to_period(Exam,P,Date,Time,Duration,Is_late,Penalty,Room),(Penalty*5)) :- exam(Exam,_,Is_large), exam_to_period(Exam,P,Date,Time,Duration,Is_late,Penalty,Room), Is_late = "yes".




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
