
% Instance Template
team(Team).

venue(Venue).

availability(Venue, Date).

% Generator

% 1
1 { assigned(Team1, Team2, Venue, Gameday) : team(Team1), team(Team2), venue(Venue), gameday(Gameday) } 1 :- team(Team1), team(Team2), venue(Venue), gameday(Gameday).


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- assigned(Team1, Team2, Venue, Gameday1), assigned(Team1, Team2, Venue, Gameday2), Gameday1 != Gameday2.


% Every team plays every other team at least once.
:- assigned(Team1, Team2, _, _), assigned(Team2, Team1, _, _).


% No team plays itself.
:- assigned(Team1, Team1, _, _).




% Soft Constraints

% Teams should play at the same venue as little as possible.
penalty("VenueStability",using_venue(Team1,T1_Venue),1) :- assigned(Team1,Team2,T1_Venue,Gameday), assigned(Team1,Team3,T2_Venue,Gameday), T1_Venue != T2_Venue.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
