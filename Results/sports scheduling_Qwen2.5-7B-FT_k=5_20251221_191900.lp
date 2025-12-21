
% Instance Template
team(Team).

venue(Venue).

availability(Venue, Date).

% Generator

% 1
1 { game(Team1, Team2, Venue, Gameday) : venue(Venue), gameday(Gameday) } 1 :- team(Team1), team(Team2).


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- game(Team1, Team2, _, Gameday1), game(Team1, Team2, _, Gameday2), Gameday1 != Gameday2.


% Every team plays every other team at least once.
game(Team1, Team2, _, _) | game(Team2, Team1, _, _) :- team(Team1), team(Team2), Team1 < Team2.


% No team plays itself.
:- game(Team, Team, _, _).




% Soft Constraints

% Teams should play at the same venue as little as possible.
penalty("VenueSwitches",game(Team1,Team2,Venue1,Gameday),1) :- game(Team1,Team2,Venue1,Gameday), game(Team1,Team2,Venue2,Gameday2), Venue1 != Venue2, Gameday1 = Gameday2.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
