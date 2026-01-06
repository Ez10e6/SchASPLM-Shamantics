
nurse(Nurse).

day(Day).

shift_type(Shift_type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Min, Max).

shift_requirement(Shift_type, Min, Max, Preferred).

% Generator

1 { assigned(Nurse, Shift, DDay) : shift_type(Shift, _), day(DDay) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift, _), day(DDay), #count{Nurse : assigned(Nurse, Shift, DDay)} < Min, nurse_requirement(Shift, Min, _).
:- shift_type(Shift, _), day(DDay), #count{Nurse : assigned(Nurse, Shift, DDay)} > Max, nurse_requirement(Shift, _, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), work_requirement(Min, Max), #count{Shift, DDay: assigned(Nurse, Shift, DDay), shift_type(Shift, Duration)} < Min.
:- nurse(Nurse), work_requirement(Min, Max), #count{Shift, DDay: assigned(Nurse, Shift, DDay), shift_type(Shift, Duration)} > Max.


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{DDay : assigned(Nurse, vacation, DDay)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, night, DDay), assigned(Nurse, morning, DDay+1).
:- assigned(Nurse, night, DDay), assigned(Nurse, afternoon, DDay+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, "afternoon", DDay), assigned(Nurse, "morning", DDay+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(Nurse), work_requirement(Min, Max), Min > 2, #count{DDay : assigned(Nurse, "rest", DDay), day(DDay)} < Min.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(Nurse, night, DDay), assigned(Nurse, night, DDay+1).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), work_requirement(Min, Max), #sum{Hours, Shift, DDay: assigned(Nurse, Shift, DDay), shift_type(Shift, Duration) } < Min.
:- nurse(Nurse), work_requirement(Min, Max), #sum{Hours, Shift, DDay: assigned(Nurse, Shift, DDay), shift_type(Shift, Duration) } > Max.




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
penalty("ShiftRequirement", shift_requirement(Shift_type, Min, Max, Preferred), (Preferred - Count)*1) :- shift_requirement(Shift_type, Min, Max, Preferred), Count = #count{Nurse, Shift, DDay : assigned(Nurse, Shift, DDay), shift_type(Shift, _), day(DDay)}.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
