
:- selected_player("Streamer1"), selected_speaker(S), A != "CaseA".

% Generator

:- pulse_at(_,T1), at("NodeX",T), T < 95.


% Hard Constraints

% No student sits more than one examination at the same time.
edge(1..0).


% The number of students taking exams in a room at the same time should not exceed the capacity of the room.
room(1..0).


% Period Lengths are not violated.
edge("SubstationA"; "UI").
edge("Junction10"; "SubstationB"; "FusionDrive").


% Some exams must be before other exams.
edge("morning"; "afternoon").
1 { topicOf(P, T) : topic(T) } 1 :- edge(L1, _), topicOf(L2, "Math").


% Some exams must be at the same time as other exams.
edge("SubstationA"; "SubstationB").
edge("Junction10", 80; "Litero", 60).


% Some exams must be at a differen time than other exams.
edge("Substation"; "MathB").
edge("Qu", 60).


% Some exams must take place in a room with no other exams.
edge("Substation"; "MathB").
edge("Qu", 60).




% Soft Constraints

% Students should not have more than one exam in the same day
:- scheduled(P1, _, V), topicOf(L1, "Math"), topicOf(W2, "Math").


% Exams of different lengths should not be scheduled in the same room in the same period
:- sampled("StationC",T), at(L2, T1), at("MidRelay",T3), T < 12.


% Some periods have associated penalties
max_timeatt("B"; "afternoon").
edge("Station1", 6).


% Some rooms have associated penalties
room(1..0).


% Students should not have more than one exam within a spread of 7 periods.
:- selected_cpu("i8"), selected_mobo(M), M != "Z690".


% An exam that is large should not be scheduled in a period that is late.
:- scheduled(P1, _, V), topicOf(L1, "Math"), topicOf(W2, "Math").




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
