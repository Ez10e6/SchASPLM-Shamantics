
team(Team).

venue(Venue).

availability(Venue, Date).

% Generator

1 { game(team1, team2, venue, gameday) : team(Team1), team(Team2), venue(Venue), gameday(Gameday) }.


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- game(team1, team2, venue1, gameday1), game(team1, team2, venue2, gameday2).


% Every team plays every other team at least once.
:- team(Team1), team(Team2), Team1 != Team2, not game(Team1, Team2, _, _).


% No team plays itself.
:- game(Team1, Team2, _, _), Team1 == Team2.




% Soft Constraints

% Teams should play at the same venue as little as possible.
penalty("VenueChange",game(team1,team2,Venue1,Gameday),1) :- game(team1,team2,Venue1,Gameday), game(team1,team2,Venue2,Gameday), Venue1 != Venue2.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
