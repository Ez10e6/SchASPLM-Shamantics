
nurse(Nurse).

day(Day).

shift_type(Shift_Type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(Shift_Type, Min, Max, Preferred).

% Generator

1 { works(Nurse, Shift_Type, DDay) : day(DDay) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift_Type, _), day(DDay), #count{Nurse : works(Nurse, Shift_Type, DDay)} < Min, nurse_requirement(Shift_Type, Min, _).
:- shift_type(Shift_Type, _), day(DDay), #count{Nurse : works(Nurse, Shift_Type, DDay)} > Max, nurse_requirement(Shift_Type, _, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #sum{Duration, Shift_Type, DDay: works(Nurse, Shift_Type, DDay), shift_type(Shift_Type, Duration)} < Min, work_requirement(Nurse, Min, _).
:- nurse(Nurse), #sum{Duration, Shift_Type, DDay: works(Nurse, Shift_Type, DDay), shift_type(Shift_Type, Duration)} > Max, work_requirement(Nurse, _, Max).


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{DDay : works(Nurse, vacation, DDay)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- works(Nurse, night, DDay), works(Nurse, morning, DDay+1).
:- works(Nurse, night, DDay), works(Nurse, afternoon, DDay+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- works(Nurse, "afternoon", DDay), works(Nurse, "morning", DDay+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(Nurse), day(DDay), day(DDay2), DDay < DDay2, DDay2 - DDay = #const{days_in_window}, #count{DDay2 : day(DDay2), DDay2 < DDay, DDay2 + 1 < DDay2, works(Nurse, "rest", DDay2), works(Nurse, "rest", DDay2 + 1)} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- works(Nurse, night, DDay), works(Nurse, night, DDay+1), not works(Nurse, special_rest, DDay+1), not works(Nurse, rest, DDay+1).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), #sum{Hours, Shift_Type, DDay: works(Nurse, Shift_Type, DDay), shift_type(Shift_Type, Duration)} < Min, work_requirement(Nurse, Min, _).
:- nurse(Nurse), #sum{Hours, Shift_Type, DDay: works(Nurse, Shift_Type, DDay), shift_type(Shift_Type, Duration)} > Max, work_requirement(Nurse, _, Max).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
penalty("shift_preference", works(Nurse, Shift_Type, DDay), 1) :- works(Nurse, Shift_Type, DDay), shift_requirement(Shift_Type, _, Preferred, _), works(Nurse, Shift_Type, DDay) < Preferred.
penalty("shift_preference", works(Nurse, Shift_Type, DDay), 1) :- works(Nurse, Shift_Type, DDay), shift_requirement(Shift_Type, _, Preferred, _), works(Nurse, Shift_Type, DDay) > Preferred.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
