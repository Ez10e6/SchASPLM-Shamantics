
exam(1..6, 1..2).
student(1..4).
period(1..8).
room(1..3).
order_constrain(1, 2).
same_time_constraint(1, 3).
different_time_constraint(2, 4).
own_room_constraint(5).

% Generator

assign_exam(E, R, P) :- room(R), period(P).
:- assign_exam(E, _, _), not topicOfExam(E, T), T != "Math".


% Hard Constraints

% No student sits more than one examination at the same time.
:- scheduled(S1, R, P, D), scheduled(S2, R, P, D), S1 != S2, topicOfExam(S1, "Math"), topicOfExam(S2, "Math").


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
:- assign_exam(E1, _, P), assign_exam(E2, _, P), E1 != E2,
topicOfExam(E1, "Math"), topicOfExam(E2, "Math"),
roomOfExam(E1, R), roomOfExam(E2, R),
:- #count{S: student(S)} > 2, room(R), period(P).


% Period Lengths are not violated.
:- schedule(1, P), P > 8.


% Some exams must be before other exams.
:- order_constrain(E1, E2), not scheduled(E1, P1, D), not scheduled(E2, P2, D), P1 > P2.


% Some exams must be at the same time as other exams.
:- schedule(E1, _, P), schedule(E2, _, P), E1 != E2, examTopic(E1, "Math"), examTopic(E2, "Math").
:- schedule(E1, _, P), schedule(E2, _, P), E1 != E2, examTopic(E1, "Bio"), examTopic(E2, "Bio").


% Some exams must be at a differen time than other exams.
:- schedule(E1, _, P1), schedule(E2, _, P2), E1 != E2, order_constrain(P1, P2).


% Some exams must take place in a room with no other exams.
:- assign_exam(E1, R, _), assign_exam(E2, R, _), E1 != E2, topicOfExam(E1, "Math"), topicOfExam(E2, "Math").




% Soft Constraints

% Students should not have more than one exam in the same day
:- schedule(1, R, P), schedule(2, R, P), topicOfExam(1, "Math"), topicOfExam(2, "Math"), 1 < 2, 2 - 1 > 1.
:- schedule(1, R, P), schedule(2, R, P), topicOfExam(1, "Math"), topicOfExam(2, "Math"), 1 < 2, 2 - 1 = 1.
:- schedule(1, R, P), schedule(2, R, P), topicOfExam(1, "Math"), topicOfExam(2, "Math"), 1 = 2, 1 < 2.


% Exams of different lengths should not be scheduled in the same room in the same period
:- assign_exam(E1, _, P), assign_exam(E2, _, P), topicOfExam(E1, "Math"), topicOfExam(E2, "Physics"), duration(E1, D1), duration(E2, D2), D1 != D2.


% Some periods have associated penalties
:- assign_exam(E, _, P), periodPenalty(P, V), V > 0, not topicOfExam(E, "Math").
:- assign_exam(E1, _, P), assign_exam(E2, _, P), E1 != E2, topicOfExam(E1, "Math"), topicOfExam(E2, "Math"), not same_time_constraint(E1, E2).
:- assign_exam(E, R, P), R = 1, P = 1.
:- assign_exam(E, R, P), R = 2, P = 1.
:- assign_exam(E, R, P), R = 3, P = 1.
:- assign_exam(E, R, P), R = 4, P = 1.
:- assign_exam(E, R, P), R = 5, P = 1.
:- assign_exam(E, R, P), R = 6, P = 1.
:- assign_exam(E, R, P), R = 7, P = 1.
:- assign_exam(E, R, P), R = 8, P = 1.


% Some rooms have associated penalties
:- assign_exam(E, R, _), room_penalty(R, P), P > 0, E != 5.


% Students should not have more than one exam within a spread of 7 periods.
:- schedule(S1, _, P1), schedule(S2, _, P2), S1 != S2,
topicOfExam(S1, "Bio"), topicOfExam(S2, "Bio"),
max(P1, P2) - min(P1, P2) <= 7, countExam(S1) = 2.


% An exam that is large should not be scheduled in a period that is late.
:- assign_exam(E, _, P), sizeOfExam(E, "Large"), P > 5, P < 8.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
