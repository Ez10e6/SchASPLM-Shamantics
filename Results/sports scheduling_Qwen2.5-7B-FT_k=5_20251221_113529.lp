
% Teams: a set of teams that plays in the league
team("TeamA"; "TeamB"; "TeamC").

% Venues: a set of venues that is available
venue("StadiumX"; "StadiumY"; "StadiumZ").

% Availabilities: A set of dates at which a given venue is available
availability("StadiumX", 1).
availability("StadiumY", 2).
availability("StadiumZ", 3).
availability("StadiumX", 4).
availability("StadiumY", 5).
availability("StadiumZ", 6).

% Generator

% Teams: a set of teams that plays in the league
team("TeamA"; "TeamB"; "TeamC").

% Venues: a set of venues that is available
venue("StadiumX"; "StadiumY"; "StadiumZ").

% Availabilities: A set of dates at which a given venue is available
availability("StadiumX", 1).
availability("StadiumY", 2).
availability("StadiumZ", 3).
availability("StadiumX", 4).
availability("StadiumY", 5).
availability("StadiumZ", 6).

% Games: a set of games
game("Game1"; "Game2"; "Game3").

% Assign each game to a pair of teams, a venue, and a game day
{{ assigned(Game, Team1, Team2, Venue, GameDay) : team(Team1), team(Team2), venue(Venue), gameday(GameDay) }} :- game(Game).


% No two games can be played at the same venue on the same day
:- assigned(Game1, _, _, Venue, GameDay), assigned(Game2, _, _, Venue, GameDay), Game1 != Game2.

% Each team must play exactly one game on their home field
home_field("TeamA", "StadiumX").
home_field("TeamB", "StadiumY").
home_field("TeamC", "StadiumZ").

:- assigned(Game, Team, _, Venue, _), home_field(Team, Venue), not assigned(_, Team, _, _, _).

% TeamA must play Game1
:- assigned(Game, "TeamA", _, _, _), Game != "Game1".

% TeamB cannot play Game2
:- assigned(Game, "TeamB", _, _, _), Game == "Game2".

% Game3 must be played on day 6
:- assigned(Game, _, _, _, Day), Game == "Game3", Day != 6.

% Game1 and Game2 must not be on the same day
:- assigned(Game1, _, _, _, GameDay), assigned(Game2, _, _, _, GameDay), Game1 != Game2.

% TeamA and TeamB cannot play against each other
:- assigned(Game, "TeamA", "TeamB", _, _).

% TeamC must play in every available venue
:- team("TeamC"), venue(Venue), not assigned("TeamC", _, _, Venue, _).

%


% Hard Constraints

% No team plays each other more than once at different gamedays.
:- assigned(Game1, Team1, Team2, _, GameDay), assigned(Game2, Team1, Team2, _, GameDay), Game1 != Game2.


% Every team plays every other team at least once.
1 { played(Team1, Team2) : team(Team1), team(Team2) } 1 :- team(Team1), team(Team2), Team1 != Team2.


% No team plays itself.
:- assigned(Game, Team, Team, _).




% Soft Constraints

% Teams should play at the same venue as little as possible.
useVenue(Team1, Team2, Venue, GameDay) :- assigned(Team1, _, Venue, GameDay), assigned(Team2, _, Venue, GameDay).
penalty("SameVenue", useVenue(Team1, Team2, Venue, _), 1) :- team(Team1), team(Team2), Venue != "Neutral".




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
