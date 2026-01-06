
team(Team).

venue(Venue).

availability(Venue, Date).

% Generator

1 { game(Team1, Team2, Venue, Gameday) : team(Team1), team(Team2), venue(Venue), gameday(Gameday) } 1 :- team(Team1), team(Team2).


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- game(Team1, Team2, _, Gameday1), game(Team1, Team2, _, Gameday2), Gameday1 != Gameday2.


% Every team plays every other team at least once.
:- team(Team1), team(Team2), not game(Team1, Team2, _, _).


% No team plays itself.
:- game(Team, Team, _, _).




% Soft Constraints

% Teams should play at the same venue as little as possible.
penalty("VenueStability",using_venue(Team,Venue),2) :- game(Team1,Team2,Venue,Gameday), game(Team1,Team3,Venue,Gameday), Team2 != Team3.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
