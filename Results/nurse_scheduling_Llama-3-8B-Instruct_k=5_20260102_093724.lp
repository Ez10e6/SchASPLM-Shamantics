
% Instance Template
nurse(Nurse).

day(0 .. N_days-1).

shift_type(Shift_type, Duration).

nurse_requirement(Shift_type, Min, Max).

work_requirement(Min, Max).

shift_requirement(Shift_type, Min, Max, Preferred).

% Generator

% 1
1 { assigned(Nurse, Shift, Day) : nurse(Nurse), day(Day) } 1 :- shift_type(Shift).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- shift_type(Shift_type, _), #count{Nurse : assigned(Nurse, Shift_type, _)} < Min, shift_type(Shift_type, _), #count{Nurse : assigned(Nurse, Shift_type, _)} > Max, nurse_requirement(Shift_type, Min, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
:- nurse(Nurse), #sum{Hours, Day: assigned(Nurse, _, Day), shift_type(_, Duration), work_requirement(Min, Max)} < Min, nurse_requirement(_, Min, _).
:- nurse(Nurse), #sum{Hours, Day: assigned(Nurse, _, Day), shift_type(_, Duration), work_requirement(Min, Max)} > Max, nurse_requirement(_, _, Max).


% Nurses must have exactly 30 days of vacation.
:- nurse(Nurse), #count{Day : day(Day), assigned(Nurse, _, Day)} < 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, shift_type(morning, _), Day), assigned(Nurse, shift_type(night, _), Day+1).
:- assigned(Nurse, shift_type(afternoon, _), Day), assigned(Nurse, shift_type(night, _), Day+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, afternoon, Day), assigned(Nurse, morning, Day+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
:- nurse(Nurse), #count{Day: day(Day), assigned(Nurse, _, Day)} < 12, #count{Day: day(Day), assigned(Nurse, 'rest', Day)} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
Corrected code:
assigned(Nurse, special_rest, Day2) :- assigned(Nurse, night, Day1), assigned(Nurse, night, Day2), Day2 =:= Day1 + 1.


% The total number of hours worked by each nurse must be between the specified minimum and maximum
Corrected code:
:- nurse(Nurse), 
   C = #sum{ { Hours } : (assigned(Nurse, Shift, Day), 
             shift_type(Shift, Duration), 
             work_requirement(Min, Max), 
             day(Day), 
             Duration * 1 =:= Hours), 
             Day }, 
   C < Min, 
   work_requirement(Min, Max).
Corrected code:
:- nurse(Nurse), 
   Count = #sum{ {Hours} : (assigned(Nurse, Shift, Day), 
                shift_type(Shift, Duration), 
                work_requirement(Min, Max), 
                day(Day), 
                Duration * 1 =:= Hours), 
             true }, 
   Count > Max, 
   work_requirement(Min, Max).




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
shift_count(Count, Shift_type, Nurse, Day) :- nurse(Nurse), day(Day), #count{Shift : assigned(Nurse, Shift, Day), shift_type(Shift, _), shift_type(Shift_type, _)} = Count.
penalty("ShiftCount", shift_count(Count, Shift_type, Nurse, Day), (Count-Preferred)*1) :- shift_count(Count, Shift_type, Nurse, Day), shift_requirement(Shift_type, _, _, Preferred).
penalty("ShiftCount", shift_count(Count, Shift_type, Nurse, Day), (Max-Count)*1) :- shift_count(Count, Shift_type, Nurse, Day), shift_requirement(Shift_type, _, Max, _).




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
