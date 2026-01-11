
nurse(Nurse).

day(Day).

shift_type(Shift_type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(Shift_type, Min, Max, Preferred).

% Generator

1 { works(Nurse, Shift_type, DDay) : day(DDay) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift_type, _), day(DDay), #count{Nurse : works(Nurse, Shift_type, DDay)} < Min, nurse_requirement(Shift_type, Min, _).
:- shift_type(Shift_type, _), day(DDay), #count{Nurse : works(Nurse, Shift_type, DDay)} > Max, nurse_requirement(Shift_type, _, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #sum{Duration, Shift_type, DDay: works(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration)} < Min, work_requirement(Nurse, Min, _).
:- nurse(Nurse), #sum{Duration, Shift_type, DDay: works(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration)} > Max, work_requirement(Nurse, _, Max).


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{DDay : works(Nurse, vacation, DDay)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- works(Nurse, night, DDay), works(Nurse, morning, DDay+1).
:- works(Nurse, night, DDay), works(Nurse, afternoon, DDay+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- works(Nurse, "afternoon", DDay), works(Nurse, "morning", DDay + 1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(Nurse), N = #count{DDay : day(DDay), works(Nurse, "rest", DDay)}, N < 2, N = #count{DDay : day(DDay), works(Nurse, "rest", DDay), DDay < #max{DDay : day(DDay)}}, N = #count{DDay : day(DDay), works(Nurse, "rest", DDay), DDay > #min{DDay : day(DDay)}}.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- works(Nurse, night, DDay), works(Nurse, night, DDay+1), not works(Nurse, special_rest, DDay+1).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), #sum{Hours, Shift_type, DDay: works(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration)} < Min, work_requirement(Nurse, Min, Max).
:- nurse(Nurse), #sum{Hours, Shift_type, DDay: works(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration)} > Max, work_requirement(Nurse, Min, Max).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
penalty("shift_preference", works(Nurse, Shift_type, DDay), 1) :- works(Nurse, Shift_type, DDay), shift_requirement(Shift_type, Min, Max, Preferred), Preferred != 0, Preferred < Min.
penalty("shift_preference", works(Nurse, Shift_type, DDay), 1) :- works(Nurse, Shift_type, DDay), shift_requirement(Shift_type, Min, Max, Preferred), Preferred != 0, Preferred > Max.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
