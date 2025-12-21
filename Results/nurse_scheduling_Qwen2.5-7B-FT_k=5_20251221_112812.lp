
% Instance Template
nurse(Nurse).

day(Day).

shift_type(ShiftType, Duration).

nurse_requirement(Shift, MinNurses, MaxNurses).

work_requirement(Nurse, MinHours, MaxHours).

shift_requirement(ShiftType, MinTimes, MaxTimes, PreferredTimes).

% Generator

% Instance Template
nurse(Nurse).

day(Day).

shift_type(ShiftType, Duration).

nurse_requirement(Shift, MinNurses, MaxNurses).

work_requirement(Nurse, MinHours, MaxHours).

shift_requirement(ShiftType, MinTimes, MaxTimes, PreferredTimes).
% Generator
1 { assigned(Nurse, Shift, Day) : shift(Shift), day(Day) } 1 :- nurse(Nurse).


% Hard Constraints

% Every day, the number of nurses assigned to each shift must be between the specified minimum and maximum
:- assigned(Nurse, Shift, Day), nurse_requirement(Shift, MinNurses, MaxNurses), count(OtherNurse, Shift, Day) < MinNurses.
count(OtherNurse, Shift, Day) :- assigned(OtherNurse, Shift, Day), not assigned(Nurse, Shift, Day).

:- assigned(Nurse, Shift, Day), nurse_requirement(Shift, MinNurses, MaxNurses), count(OtherNurse, Shift, Day) > MaxNurses.
count(OtherNurse, Shift, Day) :- assigned(OtherNurse, Shift, Day), not assigned(Nurse, Shift, Day).


% Each nurse must work at least the specified minimum and at most the specified maximum number of hours
worked(N, Hours) :- nurse(N), assigned(N, _, Day), shift_duration(Shift, Duration), Day = Day.
:- worked(N, Hours), work_requirement(N, MinHours, MaxHours), Hours < MinHours.
:- worked(N, Hours), work_requirement(N, MinHours, MaxHours), Hours > MaxHours.

worked(N, Hours) :- nurse(N), assigned(N, _, Day), shift_duration(Shift, Duration), Day = Day.
:- worked(N, Hours), work_requirement(N, MinHours, MaxHours), Hours > MaxHours.
:- worked(N, Hours), work_requirement(N, MinHours, MaxHours), Hours < MinHours.


% Nurses must have exactly 30 days of vacation.
nurse(N) :- vacation(N, _).
:- nurse(N), not vacation(N, _).
:- nurse(N), #count{D: vacation(N, D)} != 30.


% If a nurse works a night shift, they can not work a morning or afternoon shift the next day
:- assigned(Nurse, "night", Day), assigned(Nurse, "morning", Day+1).
:- assigned(Nurse, "night", Day), assigned(Nurse, "afternoon", Day+1).


% If a nurse works an afternoon shift, they cannot work a morning shift the next day.
:- assigned(Nurse, "afternoon", Day), assigned(Nurse, "morning", Day+1).


% Each nurse has at least two ordinary rest days for every window of fourteen days
rest("Ordinary").
day(D).
:- nurse(N), rest(R), R != "Ordinary", #count{D: day(D)} = X.
:- nurse(N), rest(R), R = "Ordinary", #count{D: day(D)} = Y.
:- nurse(N), #min{X: X = #count{D: day(D)}, rest(R), R != "Ordinary"}, Y < 2.

rest("Ordinary").

window(0..13).

#count{D: window(D)} = 14.

:- window(W), #count{D: day(D), W = W, rest(R), R = "Ordinary"} < 2.


% Nurses working on two consecutive nights deserve one special_rest day in addition to the ordinary rest days
:- assigned(Nurse, "night", Day1), assigned(Nurse, "night", Day2), Day2 = Day1 + 2.


% The total number of hours worked by each nurse must be between the specified minimum and maximum
:- nurse(N), time(D, T, H), assigned(N, T, D) #sum{H : assigned(N, T, D), time(D, T, H)} < MinHours.

:- nurse(N), time(D, T, H), assigned(N, T, D) #sum{H : assigned(N, T, D), time(D, T, H)} > MaxHours.




% Soft Constraints

% Nurses work the preferred number of shifts for each shift type
preferred_shifts(Nurse, ShiftType, PreferredTimes) :- nurse(Nurse), shift_type(ShiftType, _), timeslot(Timeslot), preferred_time(ShiftType, Timeslot), assigned(Nurse, ShiftType, Timeslot).
1 { assigned(Nurse, ShiftType, Timeslot) : timeslot(Timeslot), preferred_time(ShiftType, Timeslot) } PreferredTimes :- nurse(Nurse), shift_type(ShiftType, _), timeslot(Timeslot), preferred_shifts(Nurse, ShiftType, PreferredTimes).
penalty("PreferredShifts", assigned(Nurse, ShiftType, Timeslot), 1) :- nurse(Nurse), shift_type(ShiftType, _), timeslot(Timeslot), preferred_shifts(Nurse, ShiftType, PreferredTimes), #count{A: assigned(Nurse, ShiftType, Timeslot), preferred_time(ShiftType, Timeslot)} < PreferredTimes.
penalty("PreferredShifts", assigned(Nurse, ShiftType, Timeslot), 1) :- nurse(Nurse), shift_type(ShiftType, _), timeslot(Timeslot), preferred_shifts(Nurse, ShiftType, PreferredTimes), #count{A: assigned(Nurse, ShiftType, Timeslot), preferred_time(ShiftType, Timeslot)} > PreferredTimes.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
