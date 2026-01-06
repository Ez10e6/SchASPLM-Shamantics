
% Instance Template
nurse(Nurse).

day(0 .. N_days-1).

shift_type(Shift_type, Duration).

nurse_requirement(Shift_type, Min, Max).

work_requirement(Min, Max).

shift_requirement(Shift_type, Min, Max, Preferred).

% Generator

1 { assigned(Nurse, Shift, Day) : nurse(Nurse), shift_type(Shift, _), day(Day) } 1 :- nurse(Nurse), day(Day).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift, _), #count{Nurse : assigned(Nurse, Shift, _)} < Min, shift_type(Shift, _), #count{Nurse : assigned(Nurse, Shift, _)} > Max.


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #sum{Shift, Day: assigned(Nurse, Shift, Day), shift_type(Shift, Duration), day(Day)} < Min, work_requirement(Min, _).
:- nurse(Nurse), #sum{Shift, Day: assigned(Nurse, Shift, Day), shift_type(Shift, Duration), day(Day)} > Max, work_requirement(_, Max).


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{Day : assigned(Nurse, _, Day)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, Shift1, Day1), assigned(Nurse, Shift2, Day2), shift_type(Shift1, _), shift_type(Shift2, _), Shift1 = night, Shift2 = morning, Day2 = Day1+1.
:- assigned(Nurse, Shift1, Day1), assigned(Nurse, Shift2, Day2), shift_type(Shift1, _), shift_type(Shift2, _), Shift1 = night, Shift2 = afternoon, Day2 = Day1+1.
:- assigned(Nurse, Shift1, Day1), assigned(Nurse, Shift2, Day2), shift_type(Shift1, _), shift_type(Shift2, _), Shift1 = night, Shift2 = morning, Day2 = Day1+1.
:- assigned(Nurse, Shift1, Day1), assigned(Nurse, Shift2, Day2), shift_type(Shift1, _), shift_type(Shift2, _), Shift1 = night, Shift2 = afternoon, Day2 = Day1+1.


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, Shift1, Day1), assigned(Nurse, Shift2, Day2), shift_type(Shift1, afternoon), shift_type(Shift2, morning), Day2 = Day1+1.


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- day(Day), #count{Shift : assigned(_, Shift, Day), shift_type(Shift, _), shift_type(Shift, Duration), Duration != "rest", Duration != "special_rest", Duration != "vacation"} >= 12, #count{Shift : assigned(_, Shift, Day), shift_type(Shift, _), shift_type(Shift, Duration), Duration == "rest"} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(Nurse, Shift, Day1), assigned(Nurse, Shift, Day2), shift_type(Shift, Duration), Duration = night, Duration = night, Day1 != Day2, Day2 - Day1 = 1, assigned(Nurse, special_rest, _).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
Corrected code:
:- nurse(N), work_requirement(Min, Max), H = #sum{ Hours : assigned(N, _, _) }, H > Max.
Corrected code:
:- nurse(N), work_requirement(Min, Max), H = #sum{ Hours : assigned(N, _, _) }, H =< Min.




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift_requirement(Shift_type, Min, Max, Preferred) :- shift_type(Shift_type, _), nurse_requirement(Shift_type, Min, Max), Preferred = Max.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
