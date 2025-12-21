
% Instance Template
nurse(N).

day(0 .. N_days-1).

shift_type("morning", 8).
shift_type("afternoon", 8).
shift_type("night", 12).

nurse_requirement("morning", 3, 6).
nurse_requirement("afternoon", 2, 5).
nurse_requirement("night", 1, 3).

work_requirement(4, 12).

shift_requirement("morning", 3, 6, 5).
shift_requirement("afternoon", 2, 5, 4).
shift_requirement("night", 1, 3, 2).

% Generator

% 1
days(N) :- nurse(N), { assigned(N, Sh, D) : shift(Sh), day(D) }.


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
shift(Sh), day(D), #count{N : assigned(N, Sh, D)} >= Min, #count{N : assigned(N, Sh, D)} <= Max, shift_requirement(_, Sh, Min, Max).
:- shift(Sh), day(D), #count{N : assigned(N, Sh, D)} > Max, shift_requirement(_, Sh, _, Max).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
nurse_min_hours(N) :- H = #min{H1 : assigned(N, _, D), shift(_), time(D, _, H1)}, H >= MinHours(N), H <= MaxHours(N).
:- nurse(N), H = #max{H1 : assigned(N, _, D), shift(_), time(D, _, H1)}, H <= MaxHours(N), nurse_requirement(N, MinHours(N), MaxHours(N)) = true.


% Nurses must have exactly 30 days of vacation.
Corrected Code:
nurse(N) :- nurse_vacation(N).
nurse_vacation(N) :- #count{D : day(D), assigned(N, _, D)} = 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(N, Sh1, D), assigned(N, Sh2, D+1), Sh1 = "night", Sh2 != "morning", Sh2 != "afternoon".


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(N, "afternoon", D), assigned(N, "morning", D+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
Corrected Code:
:- nurse(N), #count{ D : assigned(N, _, D), shift_type("rest", T) } < 2, window(D, W), time(W).


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(N, "night", D1), assigned(N, "night", D2), D2 = D1 + 1.


% The total number of hours worked by each nurse must be between the specified minimum and maximum
sum{H : assigned(N, _, H)} >= MinHours.
sum{H : assigned(N, _, H)} <= MaxHours : nurse(N).
:- nurse(N), #sum{H : assigned(N, Sh, H), shift(Sh)} <= MaxHours.




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
work_preference("morning", 3, 6).
work_preference("afternoon", 2, 5).
work_preference("night", 1, 3).

shift_count(Count, Sh, D) :- assigned(N, Sh, D), Count = #count{ N : nurse(N) }.
penalty("WorkPreference", shift_count(Count, Sh, D), (Preferred-Count)*1) :- shift_count(Count, Sh, D), work_preference(_, Preferred, _), Count < Preferred.
penalty("WorkPreference", shift_count(Count, Sh, D), (Count-Preferred)*1) :- shift_count(Count, Sh, D), work_preference(_, Preferred, _), Count > Preferred.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
