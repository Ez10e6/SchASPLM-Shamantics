
exam(exam). student(exam, student). period(date, time, duration, is_late, penalty). room(capacity, penalty). order_constrain(exam1, exam2). same_time_constrain(exam1, exam2). different_time_constrain(exam1, exam2). own_room_constrain(exam). exam(exam), student(exam, student), duration(exam, duration), is_large(exam, is_large). period(date, time, duration, is_late, penalty), date(date), time(time), duration(duration), is_late(is_late), penalty(penalty). room(capacity, penalty), capacity(capacity), penalty(penalty). order_constrain(exam1, exam2), exam(exam1), exam(exam2). same_time_constrain(exam1, exam2), exam(exam1), exam(exam2). different_time_constrain(exam1, exam2), exam(exam1), exam(exam2). own_room_constrain(exam), exam(exam).

% Generator

% 1
Corrected Code:
1 { assigned(Exam, Period, Room) : exam(Exam), period(Period), room(Room) }.


% Hard Constraints

% No student sits more than one examination at the same time.
:- assigned(Exam1, Period1, Room), assigned(Exam2, Period1, Room), student(Exam1, Student), student(Exam2, Student).


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
Corrected Code:
:- room(Room, Capacity), #count{ Student : student(Exam, Student), assigned(Exam, _, Room) } <= Capacity.


% Period Lengths are not violated.
:- period(Period1, _, _, _, _), period(Period2, _, _, _, _), duration(Exam, Duration), Period1 + Duration = Period2.


% Some exams must be before other exams.
:- order_constrain(Exam1, Exam2), assigned(Exam1, Period1, Room), assigned(Exam2, Period2, Room), Period1 < Period2.


% Some exams must be at the same time as other exams.
:- same_time_constrain(Exam1, Exam2), assigned(Exam1, Period, Room), assigned(Exam2, Period, Room).


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, Period1, Room), assigned(Exam2, Period2, Room), Exam1!= Exam2, Period1 = Period2.


% Some exams must take place in a room with no other exams.
:- assigned(Exam1, Room, Period), assigned(Exam2, Room, Period), Exam1!= Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
penalty("SameDayExams",same_day_exams(Exam1,Exam2),(same_day_exams_consecutive(Exam1,Exam2)*7+~same_day_exams_consecutive(Exam1,Exam2)*5)) :- exam(Exam1), exam(Exam2), same_day_exams(Exam1,Exam2).
same_day_exams(Exam1,Exam2) :- assigned(Exam1,Period1,Room), assigned(Exam2,Period2,Room), date(Period1,_,_,_,_), date(Period2,_,_,_,_), Period1 = Period2.
same_day_exams_consecutive(Exam1,Exam2) :- assigned(Exam1,Period1,Room), assigned(Exam2,Period2,Room), date(Period1,_,_,_,_), date(Period2,_,_,_,_), Period1 = Period2, abs(Period1-Period2) = 1.


% Exams of different lengths should not be scheduled in the same room in the same period
Corrected Code:
penalty(10) :- assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), exam(Exam1), exam(Exam2), duration(Exam1, D1), duration(Exam2, D2), D1!= D2.


% Some periods have associated penalties
penalty("PeriodPenalty", assigned(Exam, Period, Room), penalty*1) :- assigned(Exam, Period, Room), period(Period, _, _, _, penalty).


% Some rooms have associated penalties
penalty("RoomPenalty", assigned(Exam, Period, Room), penalty*1) :- exam(Exam), period(Period), room(Room, penalty).


% Students should not have more than one exam within a spread of 7 periods.
Corrected Code:
penalty("ExamSpread", has_two_exams(Student, Exam1, Exam2), 3) :- 
    student(Student), 
    exam(Exam1), 
    exam(Exam2), 
    assigned(Exam1, Period1, Room), 
    assigned(Exam2, Period2, Room), 
    Period1 =< Period2, 
    Period2 - Period1 =< 7.


% An exam that is large should not be scheduled in a period that is late.
penalty("LargeExamLate",large_exam(Exam),5) :- exam(Exam), is_large(Exam,is_large), is_late(Period,is_late), assigned(Exam,Period,Room), is_large(Exam,is_large), is_late(Period,is_late).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
