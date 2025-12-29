
exam(Exam, Duration, Is_large).
student(Exam, Student).
period(Period, Date, Time, Duration, Is_late, Penalty).
room(Room, Capacity, Penalty).
order_constrain(Exam1, Exam2).
same_time_constrain(Exam1, Exam2).
different_time_constrain(Exam1, Exam2).
own_room_constrain(Exam).

% Generator

1 { assigned(Exam, Period, Room) : room(Room), period(Period) } 1 :- exam(Exam,_,_).


% Hard Constraints

% No student sits more than one examination at the same time.
:- assigned(Exam1, _, Room, Day, Period), assigned(Exam2, _, Room, Day, Period), student(Exam1, Student), student(Exam2, Student), Exam1 != Exam2.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- assigned(Exam1, Period, Room), assigned(Exam2, Period, Room), Exam1 != Exam2, #count{Student : student(Exam, Student), Exam = Exam1} > Capacity, room(Room, Capacity).


% Period Lengths are not violated.
:- assigned(Exam, Period, Room), exam(Exam, Duration), period(Period, _, _, Duration, _, _), Duration != 1.


% Some exams must be before other exams.
:- assigned(Exam1, _, _), assigned(Exam2, _, _), order_constrain(Exam1, Exam2), Exam1 != Exam2.


% Some exams must be at the same time as other exams.
:- assigned(Exam1, _, Room), assigned(Exam2, _, Room), Exam1 != Exam2.


% Some exams must be at a differen time than other exams.
:- assigned(Exam1, _, _), assigned(Exam2, _, _), order_constrain(Exam1, Exam2).


% Some exams must take place in a room with no other exams.
:- assigned(Exam1, _, Room), assigned(Exam2, _, Room), Exam1 != Exam2.




% Soft Constraints

% Students should not have more than one exam in the same day
:- scheduled(Exam1,Exam2), scheduled(Exam1,Period1), scheduled(Exam2,Period2), Exam1 != Exam2, Period1 = Period2 - 1, penalty("NonConsecutiveExams",7).


% Exams of different lengths should not be scheduled in the same room in the same period
:- scheduled(Exam1,Exam2), exam(Exam1,Duration1), exam(Exam2,Duration2), Duration1 != Duration2.


% Some periods have associated penalties
penalty("PeriodPenalty",assigned(Exam,Period,Room),Penalty*1) :- assigned(Exam,Period,Room), period(Period,_,_,_,_,Penalty).


% Some rooms have associated penalties
penalty("RoomPenalty",assigned(Exam,Period,Room),Penalty) :- assigned(Exam,Period,Room), room(Room,_,Penalty).


% Students should not have more than one exam within a spread of 7 periods.
:- scheduled(Exam1,Exam2), spread(Exam1,Exam2), Exam1 != Exam2.


% An exam that is large should not be scheduled in a period that is late.
penalty("LateLargeExams",late(Period),5) :- assigned(Exam,Period,_), exam(Exam,_,Is_large), Is_large = 1, late(Period).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
