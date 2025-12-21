
% Instance Template
nurse(Nurse).

day(Day).

shift_type(Type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(Type, Min, Max, Preferred).

% Generator

% Instance Template
nurse(Nurse).

day(Day).

shift_type(Type, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(Type, Min, Max, Preferred).

% Generator
1 { assigned(Nurse, Shift, Day) : shift_type(Shift, _), day(Day) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift, _), day(Day), #count{Nurse : assigned(Nurse, Shift, Day)} < nurse_requirement(Shift, Min, _).
:- shift_type(Shift, _), day(Day), #count{Nurse : assigned(Nurse, Shift, Day)} > nurse_requirement(Shift, _, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #count{Shift: assigned(Nurse, Shift, Hours), shift_type(Shift, Duration)} < Min, work_requirement(Nurse, Min, _).
:- nurse(Nurse), #count{Shift: assigned(Nurse, Shift, Hours), shift_type(Shift, Duration)} > Max, work_requirement(Nurse, _, Max).


% Nurses must have exactly 30 days of vacation.
:- nurse(N), #count{Day : assigned(N, vacation, Day)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, night, Day), assigned(Nurse, morning, Day+1).
:- assigned(Nurse, night, Day), assigned(Nurse, afternoon, Day+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, morning, Day), assigned(Nurse, afternoon, Day+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(N), day(D1), day(D2), #count{D : day(D), D1 < D, D < D2, assigned(N, 'rest', D)} >= 2, #max{D : day(D)} - #min{D : day(D)} = 13.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(_, night, Day), assigned(_, night, Day+1), #count{N: assigned(_, special_rest, Day)} < 1.


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), #sum{Hours : assigned(Nurse, Shift, Day), shift_type(Shift, Duration), Hours = Duration} < MinHours, work_requirement(Nurse, MinHours, _).
:- nurse(Nurse), #sum{Hours : assigned(Nurse, Shift, Day), shift_type(Shift, Duration), Hours = Duration} > MaxHours, work_requirement(Nurse, _, MaxHours).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift_requirement(Type, Min, Max, Preferred) :- shift_type(Type, _).
assigned(Nurse, Shift, Day) :- nurse(Nurse), day(Day), shift_type(Shift, _).
penalty("ShiftPreference", assigned(Nurse, Shift, Day), 1) :- assigned(Nurse, Shift, Day), shift_requirement(Shift, _, _, Preferred), #count{Day : assigned(Nurse, Shift, Day)} < Preferred.
penalty("ShiftPreference", assigned(Nurse, Shift, Day), 1) :- assigned(Nurse, Shift, Day), shift_requirement(Shift, _, _, Preferred), #count{Day : assigned(Nurse, Shift, Day)} > Preferred.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
