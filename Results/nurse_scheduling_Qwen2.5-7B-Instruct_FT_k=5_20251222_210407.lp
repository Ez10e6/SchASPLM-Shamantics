
% Instance Template
nurse(Nurse).

day(Day).

shift_type(Shift, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_assignment(Nurse, Day, Shift, Duration).

% Generator

% Instance Template
nurse(Nurse).

day(Day).

shift_type(Shift, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_assignment(Nurse, Day, Shift, Duration).


% 1
shift_assignment(Nurse, Day, Shift, Duration) :- nurse(Nurse), day(Day), shift_type(Shift, Duration).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- day(Day), shift_type(Shift, _Duration),
#count{N: shift_assignment(N, Day, Shift, _Duration)} < nurse_requirement(Shift, Min, _Max).
:- day(Day), shift_type(Shift, _Duration),
#count{N: shift_assignment(N, Day, Shift, _Duration)} > nurse_requirement(Shift, _Min, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(N), day(D), #count{S, D, H, ShiftAssignment: shift_assignment(N, D, Shift, H), shift_type(Shift, _)} < work_requirement(N, Min, _).
:- nurse(N), day(D), #count{S, D, H, ShiftAssignment: shift_assignment(N, D, Shift, H), shift_type(Shift, _)} > work_requirement(N, _, Max).


% Nurses must have exactly 30 days of vacation.
:- nurse(N), day(D), shift_assignment(N, D, "vacation", _), not_vacation(D).
:- nurse(N), day(D), not shift_assignment(N, D, "vacation", _), vacation(D).

vacation(D) :- day(D), not shift_assignment(_, D, _, _).
:- day(D), vacation(D), shift_assignment(_, D, _, _).


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- shift_assignment(Nurse, Day, night, 8), shift_assignment(Nurse, Day+1, Shift, _), Shift != night.


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- shift_assignment(Nurse, Day, afternoon, 6), shift_assignment(Nurse, Day+1, morning, 6).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(N), day(D1), day(D2), day(D3), day(D4), day(D5), day(D6), day(D7), day(D8), day(D9), day(D10), day(D11), day(D12), day(D13), day(D14),
shift_assignment(N, D1, "rest", _), shift_assignment(N, D2, "rest", _),
not shift_assignment(N, D3, "special_rest", _), not shift_assignment(N, D4, "special_rest", _),
not shift_assignment(N, D5, "morning", _), not shift_assignment(N, D6, "morning", _),
not shift_assignment(N, D7, "afternoon", _), not shift_assignment(N, D8, "afternoon", _),
not shift_assignment(N, D9, "night", _), not shift_assignment(N, D10, "night", _),
not shift_assignment(N, D11, "special_rest", _), not shift_assignment(N, D12, "special_rest", _),
not shift_assignment(N, D13, "morning", _), not shift_assignment(N, D14, "morning", _),
#count{D : shift_assignment(N, D, "rest", _)} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- shift_assignment(N, D, "night", 8), shift_assignment(N, D+1, "night", 8), rest_days(N, Rs), Rs < D.


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(N), day(D), #count{S: shift_assignment(N, D, S, Dur)} < MinHours, work_requirement(N, MinHours, _).
:- nurse(N), day(D), #count{S: shift_assignment(N, D, S, Dur)} > MaxHours, work_requirement(N, _, MaxHours).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
preferred_shift_count(Nurse, Shift, Count) :- nurse(Nurse), shift_type(Shift,_), work_requirement(Nurse, Min, Max), Min <= Count, Max >= Count.
actual_shift_count(Nurse, Shift, Count) :- nurse(Nurse), shift_type(Shift,_), #count{Day: shift_assignment(Nurse,Day,Shift,_)}=Count.
penalty("ShiftCount",preferred_shift_count(Nurse,Shift,Count),Abs(Count-ActualCount)) :- preferred_shift_count(Nurse,Shift,Count), actual_shift_count(Nurse,Shift,ActualCount), Count < ActualCount; Count > ActualCount.

penalty("ShiftCount",preferred_shift_count(Nurse,Shift,Count),Abs(Count-ActualCount)) :- preferred_shift_count(Nurse,Shift,Count), actual_shift_count(Nurse,Shift,ActualCount), Count < ActualCount; Count > ActualCount.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
