
team(0..N_teams-1).
venue(0..N_venues-1).
availability(Venue, Date).

% Generator

1 { scheduled(Team1, Team2, Venue, Gameday) : team(Team1), team(Team2), venue(Venue), gameday(Gameday) } 1.


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- scheduled(Team1, Team2, _, Gameday), scheduled(Team1, Team3, _, Gameday), Team2 != Team3.


% Every team plays every other team at least once.
:- scheduled(Team1, Team2, _), scheduled(Team3, Team4, _), Team1 != Team2, Team3 != Team4.


% No team plays itself.
:- scheduled(Team1, Team2, _, _), Team1 = Team2.




% Soft Constraints

% Teams should play at the same venue as little as possible.
penalty("VenueExclusivity",team(Team1),1) :- scheduled(Team2,Team1,Venue1,Gameday), scheduled(Team3,Team4,Venue2,Gameday), Team1 != Team2, Team1 != Team3, Team1 != Team4, Venue1 != Venue2.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
