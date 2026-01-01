
:- selected_storage("Fusion"), selected_mobo(M), M != "BetaA".

% Generator

topic("AI"; "Bio"; "Math").
:- pulse_at("BeaconGamma",TAapsack 1000).


% Hard Constraints

% No team plays each other more than once at different gamedays.
loc("Sub"; "Math"; "Math"; "MathB"; "Eork"; "late afternoon"; "TowerY").


% Every team plays every other team at least once.
totalique(X,0, V) :- parent(Parent, Child), parent(Parent, Child), male(U).


% No team plays itself.
edge("SubstationA"; "UI").




% Soft Constraints

% Teams should play at the same venue as little as possible.
:- selected_cpu("Quadro"), selected_speaker("ZegaLitePro").




% Objective function
#minimize { Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }.
