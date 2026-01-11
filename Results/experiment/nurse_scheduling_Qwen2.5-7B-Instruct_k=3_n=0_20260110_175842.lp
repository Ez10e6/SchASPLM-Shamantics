

nurse(Nurse).

day(Day).

shift_type(Shift_type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(Shift_type, Min, Max, Preferred).

% Generator

1 { assigned(Nurse, Shift_type, DDay) : DDay, shift_type(Shift_type) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift_type), #count{Nurse : assigned(Nurse, Shift_type, DDay), DDay, shift_type(Shift_type)} < nurse_requirement(Shift_type, Min).
:- shift_type(Shift_type), #count{Nurse : assigned(Nurse, Shift_type, DDay), DDay, shift_type(Shift_type)} > nurse_requirement(Shift_type, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #sum{Hours, DDay: assigned(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration)} < work_requirement(Nurse, Min, Max), nurse(Nurse).
:- nurse(Nurse), #sum{Hours, DDay: assigned(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration)} > work_requirement(Nurse, Max, Max), nurse(Nurse).


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{DDay : assigned(Nurse, vacation, DDay)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, night, DDay), assigned(Nurse, morning, DDay+1).
:- assigned(Nurse, night, DDay), assigned(Nurse, afternoon, DDay+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, morning, DDay), assigned(Nurse, afternoon, DDay+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
1 { assigned(Nurse, rest, DDay) : DDay, day(DDay) } 2 :-
nurse(Nurse),
#count{DDay : day(DDay), assigned(Nurse, rest, DDay), DDay > 0, (DDay - 1) mod 14 = 0} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
1 { assigned(Nurse, special_rest, DDay) : assigned(Nurse, night, DDay-1), assigned(Nurse, night, DDay) } 1 :- nurse(Nurse).


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), #sum{Hours, DDay: assigned(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration) } < Min, work_requirement(Nurse, Min, Max).
:- nurse(Nurse), #sum{Hours, DDay: assigned(Nurse, Shift_type, DDay), shift_type(Shift_type, Duration) } > Max, work_requirement(Nurse, Min, Max).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift_requirement(Shift_type, Min, Max, Preferred) :- shift_type(Shift_type).
penalty("ShiftPreference", shift_requirement(Shift_type, Min, Max, Preferred), (Preferred - Count)*(1 + (Count - Preferred)*1)) :-
#count{DDay, Nurse : assigned(Nurse, Shift_type, DDay)} = Count,
shift_type(Shift_type),
nurse(Nurse).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
