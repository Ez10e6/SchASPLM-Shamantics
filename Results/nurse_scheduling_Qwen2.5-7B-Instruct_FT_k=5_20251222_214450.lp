
% Instance Template
nurse(Nurse).

day(Day).

shift_type(ShiftType, Duration).

nurse_requirement(Shift, Min, Max).

work_requirement(Nurse, Min, Max).

shift_requirement(ShiftType, Min, Max, Preferred).

% Generator

% 1
1 { assigned(Nurse, ShiftType, Day) : day(Day), shift_type(ShiftType,_) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(ShiftType, _), day(Day), #count{Nurse : assigned(Nurse, ShiftType, Day)} < nurse_requirement(ShiftType, Min).
:- shift_type(ShiftType, _), day(Day), #count{Nurse : assigned(Nurse, ShiftType, Day)} > nurse_requirement(ShiftType, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #sum{Duration, ShiftType, Day : assigned(Nurse, ShiftType, Day), day(Day), shift_type(ShiftType, Duration)} < Min, work_requirement(Nurse, Min, _, _).
:- nurse(Nurse), #sum{Duration, ShiftType, Day : assigned(Nurse, ShiftType, Day), day(Day), shift_type(ShiftType, Duration)} > Max, work_requirement(Nurse, _, Max, _).


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{Day : assigned(Nurse, vacation, Day)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, night, Day), assigned(Nurse, morning, Day+1).
:- assigned(Nurse, night, Day), assigned(Nurse, afternoon, Day+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, morning, Day), assigned(Nurse, afternoon, Day+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
1 { assigned(Nurse, 'rest', Day) : day(Day) } 2 :-
nurse(Nurse),
#count{D : assigned(Nurse, 'rest', D), day(D), D >= #succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(FirstDay)))))))))))), D < #succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(#succ(FirstDay)))))))))))))} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(Nurse, night, Day1), assigned(Nurse, night, Day2), Day2 = Day1 + 1, #count{X: assigned(Nurse, special_rest, X)} < 1.


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(Nurse), #sum{ Hours : assigned(Nurse, Shift, Hours), shift_type(Shift, Duration) } < Min, work_requirement(Nurse, Min, _, _).
:- nurse(Nurse), #sum{ Hours : assigned(Nurse, Shift, Hours), shift_type(Shift, Duration) } > Max, work_requirement(Nurse, _, Max, _).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift_requirement(ShiftType, Min, Max, Preferred) :- shift_type(ShiftType,_).
work_shift_count(Count, Nurse, ShiftType, Day) :- nurse(Nurse), day(Day), shift_type(ShiftType,_), #count{Shift : assigned(Nurse, ShiftType, Day)} = Count.
penalty("ShiftPreference", work_shift_count(Count, Nurse, ShiftType, Day), (Count-Preferred)*1) :- shift_requirement(ShiftType, _, _, Preferred), work_shift_count(Count, Nurse, ShiftType, Day).
penalty("ShiftPreference", work_shift_count(Count, Nurse, ShiftType, Day), (Preferred-Count)*1) :- shift_requirement(ShiftType, _, _, Preferred), work_shift_count(Count, Nurse, ShiftType, Day).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
