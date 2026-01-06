
% Instance Template
nurse(Nurse).

day(Day).

shift_type(Shift_Type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(Shift_Type, Min, Max, Preferred).

% Generator

1 { assigned(Nurse, Shift_Type, DDay) : day(DDay) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift_Type, Duration), day(DDay), #count{Nurse : assigned(Nurse, Shift_Type, DDay)} < nurse_requirement(Shift_Type, Min, Max), nurse(Nurse).
:- shift_type(Shift_Type, Duration), day(DDay), #count{Nurse : assigned(Nurse, Shift_Type, DDay)} > nurse_requirement(Shift_Type, Min, Max), nurse(Nurse).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #sum{Hours, Shift_Type: assigned(Nurse, Shift_Type, Hours), shift_type(Shift_Type, _, Duration)} < work_requirement(Nurse, Min, Max), nurse(Nurse).
:- nurse(Nurse), #sum{Hours, Shift_Type: assigned(Nurse, Shift_Type, Hours), shift_type(Shift_Type, _, Duration)} > work_requirement(Nurse, Min, Max), nurse(Nurse).


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{DDay : assigned(Nurse, vacation, DDay)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, night, DDay), assigned(Nurse, morning, DDay+1).
:- assigned(Nurse, night, DDay), assigned(Nurse, afternoon, DDay+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, morning, DDay), assigned(Nurse, afternoon, DDay+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(Nurse), day(DDay), day(DDay2), #count{Shift_Type : shift_type(Shift_Type, 1), assigned(Nurse, Shift_Type, DDay), assigned(Nurse, Shift_Type, DDay2), DDay != DDay2} < 2, DDay2 - DDay =# 14.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(Nurse, night, DDay), assigned(Nurse, night, DDay+1), not assigned(Nurse, special_rest, DDay+1).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), #sum{Hours, Shift_Type: assigned(Nurse, Shift_Type, Hours), shift_type(Shift_Type, _, Duration)} < Min, work_requirement(Nurse, Min, _).
:- nurse(Nurse), #sum{Hours, Shift_Type: assigned(Nurse, Shift_Type, Hours), shift_type(Shift_Type, _, Duration)} > Max, work_requirement(Nurse, _, Max).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift_requirement(Shift_Type, Min, Max, Preferred).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
