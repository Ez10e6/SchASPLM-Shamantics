
% Instance Template
team(Team).

venue(Venue).

availability(Venue, Date).

% Generator

% 1
{ assigned(T, U, V, G) : team1(T), team2(U), venue(V), gameday(G) } 1.


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- assigned(T1, U1, G1), assigned(T2, U2, G2), T1 != U1, T1 == T2, U1 == U2, G1 != G2.


% Every team plays every other team at least once.
:- assigned(T1, T2, _, G), assigned(T2, T1, _, G), T1 != T2.


% No team plays itself.
:- assigned(T, T, _, _).




% Soft Constraints

% Teams should play at the same venue as little as possible.
penalty("VenueConsistency", venue_inconsistent(T, U, V, G), 1) :- assigned(T, U, V, G), assigned(T, W, X, Y), V != W.




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
