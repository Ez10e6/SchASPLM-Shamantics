
% Exams: A set of exams that need to be scheduled.
exam(E, D, IsLarge).

% Students: A set of students that are taking a set of exams.
student(S, E).

% Periods: A set of periods in which the exams can be scheduled.
period(P, Date, Time, Dur, IsLate, Penalty).

% Rooms: A set of rooms in which the exams can be scheduled.
room(R, Cap, Penalty).

% Order_Constrains: A set of constraints that specify the order in which some exams must be scheduled.
order_constrain(E1, E2).

% Same_Time_Constraints: A set of constraints that specify the exams that must be scheduled at the same time.
same_time_constrain(E1, E2).

% Different_Time_Constraints: A set of constraints that specify the exams that must not be scheduled at the same time.
different_time_constrain(E1, E2).

% Own_Room_Constraints: A set of constraints that specify the exams that must not be scheduled in a room on their own.
own_room_constrain(E, R).

% Generator

% Assign each exam to a room and period.
1 {scheduled_exam(E, R, P) : room(R), period(P)} 1 :- exam(E, _, _).


% Hard Constraints

% No student sits more than one examination at the same time.
:- scheduled_exam(E1, R, P), scheduled_exam(E2, R, P), student(S, E1), student(S, E2), E1 != E2.


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
room_capacity(R, Cap) :- room(R, Cap, _).
:- scheduled_exam(E1, R, P), scheduled_exam(E2, R, P), student(S, E1), student(S, E2), room_capacity(R, Cap), E1 != E2,
#count{S : scheduled_exam(E, R, P), student(S, E)} > Cap.


% Period Lengths are not violated.
:- scheduled_exam(_, _, P1), scheduled_exam(_, _, P2), period(P1, _, T1, Dur1, _, _), period(P2, _, T2, Dur2, _, _), T1 = T2, Dur1 != Dur2.


% Some exams must be before other exams.
:- scheduled_exam(E1, _, P1), scheduled_exam(E2, _, P2), order_constrain(E1, E2), P1 >= P2.


% Some exams must be at the same time as other exams.
:- scheduled_exam(E1, _, P), scheduled_exam(E2, _, P), same_time_constrain(E1, E2), E1 != E2.


% Some exams must be at a differen time than other exams.
:- scheduled_exam(E1, _, P), scheduled_exam(E2, _, P), different_time_constrain(E1, E2).


% Some exams must take place in a room with no other exams.
:- scheduled_exam(E1, R, P), scheduled_exam(E2, R, P), E1 != E2.




% Soft Constraints

% Students should not have more than one exam in the same day
:- scheduled_exam(E1, _, Day), scheduled_exam(E2, _, Day), E1 != E2, period(P1, _, _, _, _, _), period(P2, _, _, _, _, _), scheduled_exam(E1, _, P1), scheduled_exam(E2, _, P2), P1 != P2.

consecutive(P1, P2) :- period(P1, _, _, Dur, _, _), period(P2, _, _, Dur, _, _), P1 = P2 - 1.
penalty(7, assigned(E1, _, Day, P1), assigned(E2, _, Day, P2), consecutive(P1, P2)) :- scheduled_exam(E1, _, Day), scheduled_exam(E2, _, Day), E1 != E2, P1 != P2.

non_consecutive(P1, P2) :- period(P1, _, _, Dur, _, _), period(P2, _, _, Dur, _, _), P1 != P2, not consecutive(P1, P2).
penalty(5, assigned(E1, _, Day, P1), assigned(E2, _, Day, P2), non_consecutive(P1, P2)) :- scheduled_exam(E1, _, Day), scheduled_exam(E2, _, Day), E1 != E2, P1 != P2.


% Exams of different lengths should not be scheduled in the same room in the same period
scheduled_exam(E1, R, P) :- scheduled_exam(E2, R, P), exam(E1, _, Dur1), exam(E2, _, Dur2), Dur1 != Dur2.
penalty("DifferentLength",scheduled_exam(E1, R, P),10) :- scheduled_exam(E1, R, P), scheduled_exam(E2, R, P), exam(E1, _, Dur1), exam(E2, _, Dur2), Dur1 != Dur2.


% Some periods have associated penalties
penalty(P, S) :- scheduled_exam(_, P), period(P, _, "LatePeriod", _, _, 1).
penalty(P) :- scheduled_exam(_, P), period(P, _, "LatePeriod", _, _, _).
penalty>0.


% Some rooms have associated penalties
penalty(R, P) :- scheduled_exam(_, R, P), room(R, _, Pen), Pen > 0.

penalty(R, P) :- scheduled_exam(_, R, P), room(R, _, Pen), Pen > 0.


% Students should not have more than one exam within a spread of 7 periods.
has_exam(S, P) :- scheduled_exam(E, _, P), student(S, E).
:- student(S), has_exam(S, P1), has_exam(S, P2), P2 - P1 <= 7, P1 != P2, P2 != P1.
penalty("ExamSpread",has_exam(S, P),3) :- has_exam(S, P), has_exam(S, P2), P2 - P1 <= 7, P1 != P2, P2 != P1.


% An exam that is large should not be scheduled in a period that is late.
:- exam(E, _, IsLarge), period(P, _, _, _, IsLate, _), scheduled_exam(E, _, P), IsLarge = 1, IsLate = 1.
penalty("LateLargeExam",scheduled_exam(E,P),5) :- exam(E,_,IsLarge), period(P,_,_,_,IsLate,_), scheduled_exam(E,P), IsLarge=1, IsLate=1.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
