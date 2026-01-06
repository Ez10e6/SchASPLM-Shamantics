
% Instance Template
exam(Exam, Duration, Is_large).

student(Student, Exam).

period(Date, Time, Duration, Is_late, Penalty).

room(Room, Capacity, Penalty).

order_constrain(Exam1, Exam2).

same_time_constraint(Exam1, Exam2).

different_time_constraint(Exam1, Exam2).

own_room_constraint(Exam).

% Generator

1 { exam_to_period(Exam, Date, Time) : period(Date, Time, _, _, _) } 1 :- exam(Exam).

1 { exam_to_room(Exam, Room) : room(Room) } 1 :- exam(Exam).


% Hard Constraints

% No student sits more than one examination at the same time.
:- student(Student, Exam1), student(Student, Exam2), exam_to_period(Exam1, Date, Time), exam_to_period(Exam2, Date, Time), Exam1 != Exam2.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- exam_to_room(Exam, Room), exam_to_room(Exam2, Room), same_time_constraint(Exam, Exam2), #count{ Student : student(Student, Exam) } + #count{ Student : student(Student, Exam2) } > Capacity, room(Room, Capacity).


% Period Lengths are not violated.
:- exam_to_period(Exam, Date1, Time1), exam_to_period(Exam, Date2, Time2), Date1 != Date2, Time1 != Time2, duration(Exam, Duration), Time1 + Duration > Time2, Time2 + Duration > Time1.


% Some exams must be before other exams.
:- order_constrain(Exam1, Exam2), not exam_to_period(Exam1, Date1, Time1), exam_to_period(Exam2, Date2, Time2), Date1 > Date2, Time1 >= Time2.


% Some exams must be at the same time as other exams.
:- exam_to_period(Exam1, Date1, Time1), exam_to_period(Exam2, Date2, Time2), same_time_constraint(Exam1, Exam2), Date1 != Date2, Time1 != Time2.


% Some exams must be at a differen time than other exams.
:- exam_to_period(Exam1, Date1, Time1), exam_to_period(Exam2, Date2, Time2), different_time_constraint(Exam1, Exam2), Date1 == Date2, Time1 != Time2.


% Some exams must take place in a room with no other exams.
:- exam_to_room(Exam1, Room), exam_to_room(Exam2, Room), Exam1 != Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("ConsecutiveExams",exam_to_period(Exam1,Date,Time1),7) :- exam_to_period(Exam1,Date,Time1), exam_to_period(Exam2,Date,Time2), Time1 < Time2, Time2 - Time1 = 1, exam(Exam1), exam(Exam2), Exam1 != Exam2.
penalty("NonConsecutiveExams",exam_to_period(Exam1,Date,Time1),5) :- exam_to_period(Exam1,Date,Time1), exam_to_period(Exam2,Date,Time2), Time1 < Time2, Time2 - Time1 > 1, exam(Exam1), exam(Exam2), Exam1 != Exam2.


% Exams of different lengths should not be scheduled in the same room in the same period
penalty("DifferentLengthsInSameRoom",exam_to_period(Exam1,Date,Time),10) :- exam_to_period(Exam1,Date,Time), exam_to_period(Exam2,Date,Time), exam(Exam1,Duration1,Is_large), exam(Exam2,Duration2,Is_large), Duration1 != Duration2.


% Some periods have associated penalties
penalty("PenaltyPeriod",exam_to_period(Exam,Date,Time),P*1) :- exam_to_period(Exam,Date,Time), period(Date,Time,_,P,_).


% Some rooms have associated penalties
penalty("RoomPenalty",exam_to_room(Exam,Room),Penalty) :- exam_to_room(Exam,Room), room(Room,_,Penalty).


% Students should not have more than one exam within a spread of 7 periods.
penalty("ExamSpread",student(Student,Exam1),3) :- student(Student,Exam1), student(Student,Exam2), exam_to_period(Exam1,Date1,Time1), exam_to_period(Exam2,Date2,Time2), abs(Time1-Time2) < 7, not exam_to_period(Exam1,Date1,Time1), not exam_to_period(Exam2,Date2,Time2).


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExam",exam_to_period(Exam,Date,Time),5) :- exam(Exam,_,Is_large), Is_large = 1, period(Date,Time,_,Is_late,_), Is_late = 1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
