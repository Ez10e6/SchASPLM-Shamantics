
nurse(Nurse).

day(Day).

shift_type(Shift_type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(Shift_type, Min, Max, Preferred).

% Generator

1 { assigned(Nurse, Shift, DDay) : shift_type(Shift,_) } 1 :- nurse(Nurse), day(DDay).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift_type,_), day(DDay), #count{Nurse : assigned(Nurse, Shift_type, DDay)} < nurse_requirement(Shift_type,Min,Max), nurse(Nurse).
:- shift_type(Shift_type,_), day(DDay), #count{Nurse : assigned(Nurse, Shift_type, DDay)} > nurse_requirement(Shift_type,Min,Max), nurse(Nurse).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), work_requirement(Nurse, Min, Max), #sum{Hours, Shift: assigned(Nurse, Shift, Hours), shift_type(Shift, _)} < Min.
:- nurse(Nurse), work_requirement(Nurse, Min, Max), #sum{Hours, Shift: assigned(Nurse, Shift, Hours), shift_type(Shift, _)} > Max.


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{DDay : assigned(Nurse, vacation, DDay)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, night, DDay), assigned(Nurse, morning, DDay+1).
:- assigned(Nurse, night, DDay), assigned(Nurse, afternoon, DDay+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, "afternoon", DDay), assigned(Nurse, "morning", DDay+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(Nurse), day(DDay), #count{DDay2 : day(DDay2), DDay2 > DDay, DDay2 - DDay < 14, assigned(Nurse, "rest", DDay2)} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(Nurse, night, DDay), assigned(Nurse, night, DDay+1), not assigned(Nurse, special_rest, DDay+2).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), #sum{Hours, Shift, DDay: assigned(Nurse, Shift, DDay), shift_type(Shift, Duration)} < work_requirement(Nurse, Min, _).
:- nurse(Nurse), #sum{Hours, Shift, DDay: assigned(Nurse, Shift, DDay), shift_type(Shift, Duration)} > work_requirement(Nurse, _, Max).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift_requirement(Shift_type, Min, Max, Preferred).
penalty("ShiftRequirement",shift_requirement(Shift_type, Min, Max, Preferred), (Preferred-Min)*1) :- shift_requirement(Shift_type, Min, Max, Preferred), #count{Shift : assigned(Nurse, Shift, DDay), shift_type(Shift,Duration), nurse(Nurse), day(DDay)} < Preferred.
penalty("ShiftRequirement",shift_requirement(Shift_type, Min, Max, Preferred), (Max-Preferred)*1) :- shift_requirement(Shift_type, Min, Max, Preferred), #count{Shift : assigned(Nurse, Shift, DDay), shift_type(Shift,Duration), nurse(Nurse), day(DDay)} > Preferred.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
