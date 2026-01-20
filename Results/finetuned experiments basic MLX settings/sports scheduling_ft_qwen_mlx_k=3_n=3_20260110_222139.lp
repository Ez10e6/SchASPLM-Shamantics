
team(Team).

venue(Venue).

availability(Venue, Date).

% Generator

1 { game(Team1, Team2, Venue, Gameday) : team(Team1), team(Team2), venue(Venue), gameday(Gameday) }.


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- game(Team1, Team2, _, Gameday), game(Team1, Team2, _, Gameday2), Gameday != Gameday2.


% Every team plays every other team at least once.
:- team(Team1), team(Team2), Team1 != Team2, N = #count { game(Team1, Team2, _, _), game(Team2, Team1, _, _) }, N = 0.


% No team plays itself.
:- game(Team, Team, _, _).




% Soft Constraints

% Teams should play at the same venue as little as possible.
penalty("VenueChange",game(Team1,Team2,Venue1,Gameday),1) :- game(Team1,Team2,Venue1,Gameday), game(Team1,Team2,Venue2,Gameday), Venue1 != Venue2.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
