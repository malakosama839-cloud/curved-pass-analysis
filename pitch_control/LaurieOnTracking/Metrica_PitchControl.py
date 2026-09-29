#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import roots_hermitenorm

import numpy as np


def approx_cos_numpy(x):
    """
    Approximates cos(x) using a 2-segment piecewise linear function.
    Fully vectorized to work on NumPy arrays, lists, or scalars.
    """
    # Convert input to numpy array just in case a list/scalar was passed
    x = np.asarray(x)

    # Wrap x to the [0, 2*pi] interval
    x_wrapped = np.mod(x, 2 * np.pi)

    # Vectorized piecewise condition using np.where:
    # np.where(condition, true_yield, false_yield)
    approx = np.where(
        x_wrapped <= np.pi,
        1 - (2 / np.pi) * x_wrapped,  # Case 1: 0 to pi
        (2 / np.pi) * x_wrapped - 3  # Case 2: pi to 2pi
    )

    return approx

def _stable_hermgauss(n):
    """
    Numerically stable Gauss-Hermite physicist's quadrature nodes and weights.

    numpy.polynomial.hermite.hermgauss breaks at n >= ~400: its internal weight
    computation overflows / underflows, producing NaN or zero weights for every
    node.  For n <= 200 we keep numpy (fast, well-tested); for n > 200 we use
    scipy's roots_hermitenorm which builds the Jacobi tridiagonal matrix via a
    more stable eigenvalue path.

    Conversion between probabilist (scipy) and physicist (numpy) conventions:
        x_phys  = x_prob / sqrt(2)
        w_phys  = w_prob / sqrt(2)          [both integrate to sqrt(pi)]

    Any node whose weight is NaN, inf, or zero (extreme nodes lost to underflow
    in even scipy's calculation) is silently dropped.  At n = 600 this leaves
    ~530 of 600 nodes, which is more than sufficient for full accuracy.

    Parameters
    ----------
    n : int   number of desired quadrature points

    Returns
    -------
    t : ndarray  quadrature nodes  (physicist's convention)
    w : ndarray  quadrature weights (physicist's convention)
    """
    if n <= 200:
        return np.polynomial.hermite.hermgauss(n)

    # scipy path for high n
    x_prob, w_prob, _ = roots_hermitenorm(n, mu=True)
    x_phys = x_prob / np.sqrt(2)
    w_phys = w_prob / np.sqrt(2)

    # Drop any nodes where the weight underflowed to 0 or became non-finite
    valid = np.isfinite(w_phys) & (w_phys > 0) & np.isfinite(x_phys)
    return x_phys[valid], w_phys[valid]


def initialise_players(team, teamname, params, GKid):
    """
    initialise_players(team,teamname,params)

    create a list of player objects that holds their positions and velocities from the tracking data dataframe

    Parameters
    -----------

    team: row (i.e. instant) of either the home or away team tracking Dataframe
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )

    Returns
    -----------

    team_players: list of player objects for the team at at given instant

    """
    # get player  ids
    player_ids = np.unique([c.split('_')[1] for c in team.keys() if c[:4] == teamname])
    # create list
    team_players = []
    for p in player_ids:
        # create a player object for player_id 'p'
        team_player = player(p, team, teamname, params, GKid)
        if team_player.inframe:
            team_players.append(team_player)
    return team_players


def check_offsides(attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):
    """
    check_offsides( attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):

    checks whetheer any of the attacking players are offside (allowing for a 'tol' margin of error). Offside players are removed from
    the 'attacking_players' list and ignored in the pitch control calculation.

    Parameters
    -----------
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_position: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        verbose: if True, print a message each time a player is found to be offside
        tol: A tolerance parameter that allows a player to be very marginally offside (up to 'tol' m) without being flagged offside. Default: 0.2m

    Returrns
    -----------
        attacking_players: list of 'player' objects for the players on the attacking team with offside players removed
    """
    # find jersey number of defending goalkeeper (just to establish attack direction)
    defending_GK_id = GK_numbers[1] if attacking_players[0].teamname == 'Home' else GK_numbers[0]
    # make sure defending goalkeeper is actually on the field!
    assert defending_GK_id in [p.id for p in
                               defending_players], "Defending goalkeeper jersey number not found in defending players"
    # get goalkeeper player object
    defending_GK = [p for p in defending_players if p.id == defending_GK_id][0]
    # use defending goalkeeper x position to figure out which half he is defending (-1: left goal, +1: right goal)
    defending_half = np.sign(defending_GK.position[0])
    # find the x-position of the second-deepest defeending player (including GK)
    second_deepest_defender_x = sorted([defending_half * p.position[0] for p in defending_players], reverse=True)[1]
    # define offside line as being the maximum of second_deepest_defender_x, ball position and half-way line
    offside_line = max(second_deepest_defender_x, defending_half * ball_position[0], 0.0) + tol
    # any attacking players with x-position greater than the offside line are offside
    if verbose:
        for p in attacking_players:
            if p.position[0] * defending_half > offside_line:
                print("player %s in %s team is offside" % (p.id, p.playername))
    attacking_players = [p for p in attacking_players if p.position[0] * defending_half <= offside_line]
    return attacking_players


class player(object):
    """
    player() class

    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player

    __init__ Parameters
    -----------
    pid: id (jersey number) of player
    team: row of tracking data for team
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )


    methods include:
    -----------
    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept

    """

    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self, pid, team, teamname, params, GKid):
        self.id = pid
        self.is_gk = self.id == GKid
        self.teamname = teamname
        self.playername = "%s_%s_" % (teamname, pid)
        self.vmax = params['max_player_speed']  # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time']  # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma']  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params['lambda_att']  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params['lambda_gk'] if self.is_gk else params[
            'lambda_def']  # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.get_position(team)
        self.get_velocity(team)
        self.PPCF = 0.  # initialise this for later

    def get_position(self, team):
        self.position = np.array([team[self.playername + 'x'], team[self.playername + 'y']])
        self.inframe = not np.any(np.isnan(self.position))

    def get_velocity(self, team):
        self.velocity = np.array([team[self.playername + 'vx'], team[self.playername + 'vy']])
        if np.any(np.isnan(self.velocity)):
            self.velocity = np.array([0., 0.])

    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0.  # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity * self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final - r_reaction) / self.vmax
        return self.time_to_intercept

    import numpy as np

    def time_to_intercept_constant_acceleration(self, target_pos, amax=1.2, reaction_time=0.7):
        # 1. Calculate the distance and direction vector to the target
        diff = target_pos - self.position
        dist = np.linalg.norm(diff)

        # Edge case: Player is already at the target
        if dist < 0.1:  # 10cm tolerance
            return 0.0

        # 2. Project current velocity onto the direction vector
        # This gives us 'v0': the speed the player is already traveling towards the target
        # If v0 is negative, the player is moving away and must decelerate/turn.
        unit_direction = diff / dist
        v0 = np.dot(self.velocity, unit_direction)

        # 3. Solve the Quadratic Equation: 0.5*a*t^2 + v0*t - d = 0
        # t = (-v0 + sqrt(v0^2 + 2*a*d)) / a
        discriminant = v0 ** 2 + 2 * amax * dist

        # If discriminant is negative (mathematically impossible here since a, d > 0), return NaN
        if discriminant < 0:
            return np.nan

        t_arrival = (-v0 + np.sqrt(discriminant)) / amax

        # 4. Add Reaction Time
        # Total time = Reaction Delay + Physical Travel Time
        total_time = t_arrival + reaction_time

        return total_time

    def probability_intercept_ball(self, T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1 / (1. + np.exp(-np.pi / np.sqrt(3.0) / self.tti_sigma * (T - self.time_to_intercept)))
        return f


""" Generate pitch control map """

def default_model_params(time_to_control_veto=3):
    """
    default_model_params()
    
    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
    
    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
    
    
    Returns
    -----------
    
    params: dictionary of parameters required to determine and calculate the model
    
    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 5. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    # --------------------------------------------------------------------
    # SPEEDUP: collapsed from 10 bins (x2 bend directions = 20 trajectories
    # per grid cell) down to 5 bins (x2 bend directions = 10 trajectories
    # per grid cell) -> ~2x fewer trajectory integrations for the
    # curve-based PPCF. curvature_value for each merged bin is the average
    # of the two original bins' curvature_value it replaces, so the
    # weighted average curvature the model sees is essentially unchanged.
    # --------------------------------------------------------------------
    params['trajectory_classes'] = [
        {
            'curvature_min': 0.000000,
            'curvature_max': 0.017272,
            'curvature_value': 0.009089,
            'probability': 0.20
        },
        {
            'curvature_min': 0.017272,
            'curvature_max': 0.035427,
            'curvature_value': 0.025986,
            'probability': 0.20
        },
        {
            'curvature_min': 0.035427,
            'curvature_max': 0.065521,
            'curvature_value': 0.049209,
            'probability': 0.20
        },
        {
            'curvature_min': 0.065521,
            'curvature_max': 0.132785,
            'curvature_value': 0.094816,
            'probability': 0.20
        },
        {
            'curvature_min': 0.132785,
            'curvature_max': 792.793752,
            'curvature_value': 0.240787,
            'probability': 0.20
        },
    ]
    return params


def generate_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, GK_numbers,
                                     field_dimen=(106., 68.,), n_grid_cells_x=50, offsides=True):
    """ generate_pitch_control_for_event

    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)

    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.

    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')

    Returrns
    -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """
    # get the details of the event (frame, team in possession, ball_start_position)
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]['Start X'], events.loc[event_id]['Start Y']])
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2. + dx / 2.
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2. + dy / 2.
    # initialise pitch control grids for attacking and defending teams
    PPCFa = np.zeros(shape=(len(ygrid), len(xgrid)))
    PPCFd = np.zeros(shape=(len(ygrid), len(xgrid)))
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if pass_team == 'Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        defending_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    elif pass_team == 'Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        attacking_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    else:
        assert False, "Team in possession must be either home or away"

    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)
    # calculate pitch pitch control model at each location on the pitch
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            PPCFa[i, j], PPCFd[i, j] = calculate_pitch_control_at_target(target_position, attacking_players,
                                                                         defending_players, ball_start_pos, params)
    # check probabilitiy sums within convergence
    checksum = np.sum(PPCFa + PPCFd) / float(n_grid_cells_y * n_grid_cells_x)
    assert 1 - checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1 - checksum)
    return PPCFa, xgrid, ygrid

def generate_pitch_control_for_event_linearized_matrixform(event_id, events, tracking_home, tracking_away, params,
                                                               GK_numbers,
                                                               field_dimen=(106., 68.), n_grid_cells_x=50,
                                                               offsides=True, sigma=13.0):
    """
    Fast pitch control estimation using FastGaussianGrid influence maps.
    """
    # --- Event details ---
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]['Start X'], events.loc[event_id]['Start Y']])

    # --- Build grid ---
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2. + dx / 2.
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2. + dy / 2.

    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    # --- Initialise players ---
    if pass_team == 'Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        defending_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    elif pass_team == 'Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        attacking_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    else:
        raise ValueError("Team in possession must be either 'Home' or 'Away'")

    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)

    def get_positions(players):
        return np.array([[p.position[0], p.position[1]] for p in players])

    att_pos = get_positions(attacking_players)
    def_pos = get_positions(defending_players)

    fgg = FastGaussianGrid(sigma=38)
    att_influence = fgg.generate_heatmap(grid_coords, att_pos, sharpness=40)
    def_influence = fgg.generate_heatmap(grid_coords, def_pos, sharpness=40)

    total = att_influence + def_influence
    PPCFa_flat = np.where(total > 0, att_influence / total, 0.5)
    PPCFa = PPCFa_flat.reshape(len(ygrid), len(xgrid))
    return PPCFa, xgrid, ygrid

def generate_pitch_control_for_event_with_exponential(event_id, events, tracking_home, tracking_away, params,
                                                               GK_numbers,
                                                               field_dimen=(106., 68.), n_grid_cells_x=50,
                                                               offsides=True, sigma=2.0):
    """
    Fast pitch control estimation using FastGaussianGrid influence maps.
    """
    # --- Event details ---
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]['Start X'], events.loc[event_id]['Start Y']])

    # --- Build grid ---
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2. + dx / 2.
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2. + dy / 2.

    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    # --- Initialise players ---
    if pass_team == 'Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        defending_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    elif pass_team == 'Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        attacking_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    else:
        raise ValueError("Team in possession must be either 'Home' or 'Away'")

    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)

    def get_positions(players):
        return np.array([[p.position[0], p.position[1]] for p in players])

    att_pos = get_positions(attacking_players)
    def_pos = get_positions(defending_players)

    fgg = NonLinearGaussianGrid(sigma=sigma)
    att_influence = fgg.generate_heatmap(grid_coords, att_pos, sharpness=40)
    def_influence = fgg.generate_heatmap(grid_coords, def_pos, sharpness=40)

    total = att_influence + def_influence
    PPCFa_flat = np.where(total > 0, att_influence / total, 0.5)
    PPCFa = PPCFa_flat.reshape(len(ygrid), len(xgrid))
    return PPCFa, xgrid, ygrid


def generate_gaussian_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, GK_numbers,
                                              field_dimen=(106., 68.,), n_grid_cells_x=50, offsides=True):
    """
    Evaluates a Gaussian influence pitch control surface over the entire field.
    """
    # 1. Get the details of the event
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]['Start X'], events.loc[event_id]['Start Y']])
    
    # 2. Break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2. + dx / 2.
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2. + dy / 2.
    
    # Create a 2D meshgrid for fully vectorized coordinate calculations
    X, Y = np.meshgrid(xgrid, ygrid)
    
    # 3. Initialise player positions
    if pass_team == 'Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        defending_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    elif pass_team == 'Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        attacking_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    else:
        assert False, "Team in possession must be either home or away"

    # Find any attacking players that are offside and remove them from calculation
    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)

    # 4. Calculate Gaussian Influence maps
    # We pull the standard deviation (influence radius) from params, or default to 13.0 meters
    sigma = params.get('influence_radius', 4.0)
    
    influence_att = np.zeros_like(X)
    influence_def = np.zeros_like(X)

    # Sum the Gaussian influence for all attacking players
    for player in attacking_players:
        # Assuming the 'player' object has a '.position' attribute as per the original tutorial
        px, py = player.position[0], player.position[1]
        dist_squared = (X - px)**2 + (Y - py)**2
        influence_att += np.exp(-dist_squared / (2 * sigma**2))

    # Sum the Gaussian influence for all defending players
    for player in defending_players:
        px, py = player.position[0], player.position[1]
        dist_squared = (X - px)**2 + (Y - py)**2
        influence_def += np.exp(-dist_squared / (2 * sigma**2))

    # 5. Calculate final Pitch Control probability ratio
    # Add a tiny epsilon to the denominator to prevent division by zero in empty areas
    epsilon = 1e-8
    PPCFa = influence_att / (influence_att + influence_def + epsilon)
    
    return PPCFa, xgrid, ygrid


class GaussianLinearizer:
    """
    Approximates exp(-r) using the Gauss-Hermite / Fourier identity:

        exp(-r) = (1/√π) ∫ exp(-t²) cos(2√r · t) dt

    Quadrature discretises this as:

        exp(-r) ≈ Σᵢ cᵢ · cos(2√r · tᵢ)    where cᵢ = wᵢ / √π

    The feature vector φ(r) = [cos(2√r·t₀), …, cos(2√r·tₙ₋₁)] keeps the
    map purely linear: exp(-r) = φ(r) @ c is a single dot product.

    Fix (v2) — two causes of NaN / wrong results
    ---------------------------------------------
    CAUSE 1 — hermgauss breaks at n ≥ 380:
        numpy computes weights as 1 / H_n(tᵢ)².  At high n the polynomial
        values overflow float64, producing inf/NaN weights, which propagate
        into c and then into predict().  Fixed by switching to
        scipy.special.roots_hermitenorm (more stable Jacobi-matrix path)
        for n > 200, converting probabilist→physicist convention, and
        dropping the ~70 extreme nodes whose weights still underflow.

    CAUSE 2 — quadrature oscillation for large r_sq:
        The integral identity is exact, but the *quadrature* converges only
        when the integrand cos(2√r·t)·exp(-t²) is well-sampled by the nodes.
        Nodes concentrate in [−√(2n), √(2n)], so accurate sampling requires
        r_sq < n·π²/8 ≈ 1.23·n.  On a 106×68 m pitch with σ=5, r_sq can
        reach ~544, requiring n ≥ 442.  Below that threshold the quadrature
        oscillates around zero and can return small *negative* values.
        Those negatives make gaussian_values.sum() negative, which makes
        total = att + def near-zero or negative → NaN in the final division.
        Fixed by clamping predict() output to ≥ 0 (exp(−r) is never negative).

    Parameters
    ----------
    n_features : int
        Quadrature points.  Use at least ceil(r_sq_max · 8/π²) + margin.
        For a standard pitch with σ=5: n ≥ 500 is safe.  Up to 600 works.
    x_scale : float
        Divides r before √ so the approximation targets exp(−r/x_scale).
        Set to 1.0 (default) to approximate exp(−r) unchanged.
    """

    def __init__(self, n_features=20, x_scale=1.0):
        self.n_features = n_features
        self.x_scale = x_scale
        t_i, w_i = _stable_hermgauss(n_features)  # FIX 1: stable backend
        self.c = w_i / np.sqrt(np.pi)
        self.t = t_i

    def transform(self, x):
        """Fourier feature matrix φ(x), shape (len(x), n_features_actual)."""
        x = np.atleast_1d(x)
        if np.any(x < 0):
            raise ValueError("x must be >= 0")
        sqrt_x = np.sqrt(x.ravel() / self.x_scale)
        return np.cos(2 * np.outer(sqrt_x, self.t))

    def predict(self, x):
        """Approximate exp(-x / x_scale) via Hermite quadrature dot product."""
        phi = approx_cos_numpy(2 * np.outer(x.ravel() / self.x_scale, self.t))
        result = phi @ self.c
        return np.maximum(result, 0.0)  # FIX 2: clamp quadrature oscillation near 0


class FastGaussianGrid:
    """
    Vectorised Gaussian influence map for all players on a grid.

    Computes  Σ_p exp(−‖g−p‖²/σ²)  for every grid point g, via:

      1. 4-D hyperspace linearisation: r² = g̃ᵀ M p̃  (exact, one matmul)
      2. Fourier/Hermite approximation: exp(−r²) ≈ φ(r²) @ c  (linear)

    See GaussianLinearizer for the stability fixes applied to step 2.
    """

    def __init__(self, n_features=30, sigma=1.0):
        self.sigma = sigma
        t_i, w_i = _stable_hermgauss(n_features)
        self.c = w_i / np.sqrt(np.pi)
        self.t = t_i
        # Cache linearizer once: this class is called heavily in reconstruction loops.
        self.linearizer = GaussianLinearizer(n_features=20, x_scale=1)
        self.M = np.array([
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

    def _lift_coords(self, coords):
        """Lift 2-D [x, y] → 4-D [x/σ, y/σ, (x²+y²)/σ², 1]."""
        coords = np.atleast_2d(coords)
        x = coords[:, 0] / self.sigma
        y = coords[:, 1] / self.sigma
        norm_sq = x ** 2 + y ** 2
        return np.column_stack((x, y, norm_sq, np.ones_like(x)))

    def _aggregate_gaussian_values(self, r_sq, sharpness=40.0):
        gaussian_values = np.maximum(self.linearizer.predict(r_sq).reshape(r_sq.shape), 0.0)
        gaussian_values = gaussian_values ** sharpness
        player_weights = np.ones((gaussian_values.shape[1], 1), dtype=gaussian_values.dtype)
        return (gaussian_values @ player_weights).ravel()

    def generate_heatmap_from_lifted_grid(self, G_tilde, player_coords, sharpness=40.0):
        P_tilde = self._lift_coords(player_coords)
        r_sq = G_tilde @ self.M @ P_tilde.T
        r_sq = np.maximum(r_sq, 0.0)
        return self._aggregate_gaussian_values(r_sq, sharpness=sharpness)

    def generate_heatmap(self, grid_coords, player_coords, sharpness=40.0):
        G_tilde = self._lift_coords(grid_coords)
        return self.generate_heatmap_from_lifted_grid(G_tilde, player_coords, sharpness=sharpness)


class NonLinearGaussianGrid:
    """
    Vectorised Gaussian influence map for all players on a grid.

    Computes  Σ_p exp(−‖g−p‖²/σ²)  for every grid point g, via:

      1. 4-D hyperspace linearisation: r² = g̃ᵀ M p̃  (exact, one matmul)
      2. Fourier/Hermite approximation: exp(−r²) ≈ φ(r²) @ c  (linear)

    See GaussianLinearizer for the stability fixes applied to step 2.
    """

    def __init__(self, n_features=30, sigma=1.0):
        self.sigma = sigma
        t_i, w_i = _stable_hermgauss(n_features)
        self.c = w_i / np.sqrt(np.pi)
        self.t = t_i
        self.M = np.array([
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

    def _lift_coords(self, coords):
        """Lift 2-D [x, y] → 4-D [x/σ, y/σ, (x²+y²)/σ², 1]."""
        coords = np.atleast_2d(coords)
        x = coords[:, 0] / self.sigma
        y = coords[:, 1] / self.sigma
        norm_sq = x ** 2 + y ** 2
        return np.column_stack((x, y, norm_sq, np.ones_like(x)))

    def generate_heatmap(self, grid_coords, player_coords, sharpness=40.0):
        G_tilde = self._lift_coords(grid_coords)
        P_tilde = self._lift_coords(player_coords)

        r_sq = G_tilde @ self.M @ P_tilde.T
        r_sq = np.maximum(r_sq, 0.0)

        gaussian_values = np.exp(r_sq)

        return gaussian_values.sum(axis=1)


def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    if ball_start_pos is None or any(np.isnan(ball_start_pos)):
        ball_travel_time = 0.0
    else:
        ball_travel_time = np.linalg.norm(target_position - ball_start_pos) / params['average_ball_speed']

    tau_min_att = np.nanmin([p.simple_time_to_intercept(target_position) for p in attacking_players])
    tau_min_def = np.nanmin([p.simple_time_to_intercept(target_position) for p in defending_players])

    if tau_min_att - max(ball_travel_time, tau_min_def) >= params['time_to_control_def']:
        return 0., 1.
    elif tau_min_def - max(ball_travel_time, tau_min_att) >= params['time_to_control_att']:
        return 1., 0.
    else:
        attacking_players = [p for p in attacking_players if
                             p.time_to_intercept - tau_min_att < params['time_to_control_att']]
        defending_players = [p for p in defending_players if
                             p.time_to_intercept - tau_min_def < params['time_to_control_def']]
        dT_array = np.arange(ball_travel_time - params['int_dt'], ball_travel_time + params['max_int_time'],
                             params['int_dt'])
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)
        ptot = 0.0
        i = 1
        while 1 - ptot > params['model_converge_tol'] and i < dT_array.size:
            T = dT_array[i]
            for player in attacking_players:
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_intercept_ball(
                    T) * player.lambda_att
                assert dPPCFdT >= 0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT * params['int_dt']
                PPCFatt[i] += player.PPCF
            for player in defending_players:
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_intercept_ball(
                    T) * player.lambda_def
                assert dPPCFdT >= 0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT * params['int_dt']
                PPCFdef[i] += player.PPCF
            ptot = PPCFdef[i] + PPCFatt[i]
            i += 1
        if i >= dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot))
        return PPCFatt[i - 1], PPCFdef[i - 1]

def generate_pitch_control_for_event_curve_based(event_id, events, tracking_home, tracking_away, params, GK_numbers, field_dimen = (106.,68.,), n_grid_cells_x = 50, offsides=True):
    """ generate_pitch_control_for_event
    
    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.
        
    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')
        
    Returrns
    -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """
    # get the details of the event (frame, team in possession, ball_start_position)
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]['Start X'],events.loc[event_id]['Start Y']])
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x)*dx - field_dimen[0]/2. + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy - field_dimen[1]/2. + dy/2.
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if pass_team=='Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame],'Home',params,GK_numbers[0])
        defending_players = initialise_players(tracking_away.loc[pass_frame],'Away',params,GK_numbers[1])
    elif pass_team=='Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame],'Home',params,GK_numbers[0])
        attacking_players = initialise_players(tracking_away.loc[pass_frame],'Away',params,GK_numbers[1])
    else:
        assert False, "Team in possession must be either home or away"
        
    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        attacking_players = check_offsides( attacking_players, defending_players, ball_start_pos, GK_numbers)
    # calculate pitch pitch control model at each location on the pitch
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target_curve_based(target_position, attacking_players, defending_players, ball_start_pos, params)
    # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
    assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa,xgrid,ygrid


def generate_ball_trajectory(
    ball_start_pos,
    target_position,
    curvature,
    average_ball_speed,
    int_dt,
    bend_direction=1
):
    """
    Generate a quadratic Bezier ball trajectory from start to target.

    curvature is the normalized curvature:
        curvature = area_between_curve_and_chord / chord_length^2

    For a quadratic Bezier curve:
        area = L * h / 3

    therefore:
        h = 3 * curvature * L
    """

    start = np.asarray(ball_start_pos, dtype=float)
    end = np.asarray(target_position, dtype=float)

    chord = end - start
    chord_length = np.linalg.norm(chord)

    # Ball is already at target
    if chord_length < 1e-12:
        return np.array([start]), np.array([0.0])

    # Unit vector along start -> target
    direction = chord / chord_length

    # Perpendicular unit vector
    perpendicular = np.array([
        -direction[1],
        direction[0]
    ])

    # ------------------------------------------------------------
    # Convert normalized curvature to Bezier control-point offset
    # ------------------------------------------------------------

    h = 3.0 * curvature * chord_length

    control_point = (
        (start + end) / 2.0
        + bend_direction * h * perpendicular
    )

    # ------------------------------------------------------------
    # Dense Bezier curve used to calculate arc length
    #
    # SPEEDUP: n_dense reduced from 1000 -> 150. Arc-length only needs to
    # be accurate to sub-centimetre precision for a smooth quadratic
    # Bezier curve; 150 samples is more than enough and cuts the cost of
    # this (previously dominant) step by ~6-7x with negligible impact on
    # trajectory timing.
    # ------------------------------------------------------------

    n_dense = 150
    t_dense = np.linspace(0.0, 1.0, n_dense)

    curve_dense = (
        ((1.0 - t_dense) ** 2)[:, None] * start
        + (
            2.0
            * (1.0 - t_dense)
            * t_dense
        )[:, None] * control_point
        + (t_dense ** 2)[:, None] * end
    )

    # Segment lengths
    segment_lengths = np.linalg.norm(
        np.diff(curve_dense, axis=0),
        axis=1
    )

    # Cumulative arc length
    cumulative_distance = np.concatenate([
        [0.0],
        np.cumsum(segment_lengths)
    ])

    total_distance = cumulative_distance[-1]

    # ------------------------------------------------------------
    # Convert arc length to time
    # ------------------------------------------------------------

    total_time = total_distance / average_ball_speed

    times = np.arange(
        0.0,
        total_time,
        int_dt
    )

    # Always include final target
    if len(times) == 0 or times[-1] < total_time:
        times = np.append(times, total_time)

    # Distance travelled at each time
    target_distances = times * average_ball_speed

    # Convert distance -> Bezier parameter
    t_samples = np.interp(
        target_distances,
        cumulative_distance,
        t_dense
    )

    # Final trajectory positions
    positions = (
        ((1.0 - t_samples) ** 2)[:, None] * start
        + (
            2.0
            * (1.0 - t_samples)
            * t_samples
        )[:, None] * control_point
        + (t_samples ** 2)[:, None] * end
    )

    return positions, times


def calculate_pitch_control_for_trajectory(
    trajectory_positions,
    trajectory_times,
    attacking_players,
    defending_players,
    params
):
    """
    Calculate trajectory-dependent pitch control.

    At every integration timestep, the ball position is determined
    from the trajectory and each player's time-to-intercept is
    recalculated for that ball position.

    This allows players to intercept the ball before it reaches
    the final target.
    """

    PPCFatt = 0.0
    PPCFdef = 0.0
    ptot = 0.0

    # Integration continues beyond ball arrival so that the
    # interception probability can converge.
    max_time = (
        trajectory_times[-1]
        + params['max_int_time']
    )

    dT_array = np.arange(
        0.0,
        max_time,
        params['int_dt']
    )

    i = 0

    while (
        1.0 - ptot > params['model_converge_tol']
        and i < len(dT_array)
    ):

        T = dT_array[i]

        # --------------------------------------------------------
        # Ball position at time T
        # --------------------------------------------------------

        ball_position = np.array([
            np.interp(
                T,
                trajectory_times,
                trajectory_positions[:, dim]
            )
            for dim in range(2)
        ])

        # --------------------------------------------------------
        # Recalculate each player's arrival time to the CURRENT
        # ball position.
        # --------------------------------------------------------

        for player in attacking_players:
            player.simple_time_to_intercept(ball_position)

        for player in defending_players:
            player.simple_time_to_intercept(ball_position)

        # --------------------------------------------------------
        # Attacking team
        # --------------------------------------------------------

        for player in attacking_players:

            dPPCFdT = (
                (1.0 - PPCFatt - PPCFdef)
                * player.probability_intercept_ball(T)
                * player.lambda_att
            )

            assert dPPCFdT >= 0, (
                'Invalid attacking player probability '
                '(calculate_pitch_control_for_trajectory)'
            )

            PPCFatt += (
                dPPCFdT
                * params['int_dt']
            )

        # --------------------------------------------------------
        # Defending team
        # --------------------------------------------------------

        for player in defending_players:

            dPPCFdT = (
                (1.0 - PPCFatt - PPCFdef)
                * player.probability_intercept_ball(T)
                * player.lambda_def
            )

            assert dPPCFdT >= 0, (
                'Invalid defending player probability '
                '(calculate_pitch_control_for_trajectory)'
            )

            PPCFdef += (
                dPPCFdT
                * params['int_dt']
            )

        ptot = PPCFatt + PPCFdef

        i += 1

    if i >= len(dT_array):
        print(
            "Integration failed to converge: %1.3f"
            % ptot
        )

    return PPCFatt, PPCFdef

def calculate_pitch_control_at_target_curve_based(
    target_position,
    attacking_players,
    defending_players,
    ball_start_pos,
    params
):
    """
    Calculate pitch control probability at a target position.

    Ball arrival is modeled using multiple possible curved
    trajectories. Each trajectory is evaluated independently
    using the trajectory-dependent Spearman pitch-control model.

    The final PPCF is the probability-weighted average over
    all trajectory classes and both bending directions.
    """

    # ============================================================
    # CASE 1: Ball is already at the target
    # ============================================================

    if ball_start_pos is None or np.any(np.isnan(ball_start_pos)):

        trajectory_positions = np.array([
            target_position
        ])

        trajectory_times = np.array([
            0.0
        ])

        return calculate_pitch_control_for_trajectory(
            trajectory_positions,
            trajectory_times,
            attacking_players,
            defending_players,
            params
        )

    # ============================================================
    # CASE 2: Ball is travelling to the target
    # ============================================================

    PPCFatt = 0.0
    PPCFdef = 0.0

    probability_total = 0.0

    # ============================================================
    # Evaluate each curvature class
    # ============================================================

    for trajectory_class in params['trajectory_classes']:

        probability = trajectory_class['probability']
        curvature = trajectory_class['curvature_value']

        # --------------------------------------------------------
        # curvature_3d_normalized is unsigned.
        #
        # Therefore, initially assume equal probability of bending
        # to either side of the start -> target chord.
        # --------------------------------------------------------

        for bend_direction in [-1, 1]:

            trajectory_positions, trajectory_times = (
                generate_ball_trajectory(
                    ball_start_pos=ball_start_pos,
                    target_position=target_position,
                    curvature=curvature,
                    average_ball_speed=params['average_ball_speed'],
                    int_dt=params['int_dt'],
                    bend_direction=bend_direction
                )
            )

            # ----------------------------------------------------
            # Calculate trajectory-dependent PPCF
            # ----------------------------------------------------

            PPCFatt_k, PPCFdef_k = (
                calculate_pitch_control_for_trajectory(
                    trajectory_positions,
                    trajectory_times,
                    attacking_players,
                    defending_players,
                    params
                )
            )

            # ----------------------------------------------------
            # Probability weight
            #
            # trajectory class probability × 0.5 for each
            # bending direction.
            # ----------------------------------------------------

            weight = probability * 0.5

            PPCFatt += weight * PPCFatt_k
            PPCFdef += weight * PPCFdef_k

            probability_total += weight

    # ============================================================
    # Normalize
    # ============================================================

    if probability_total > 0:

        PPCFatt /= probability_total
        PPCFdef /= probability_total

    # Numerical safety
    PPCFatt = np.clip(
        PPCFatt,
        0.0,
        1.0
    )

    PPCFdef = np.clip(
        PPCFdef,
        0.0,
        1.0
    )

    return PPCFatt, PPCFdef