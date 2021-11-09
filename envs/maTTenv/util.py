import numpy as np 
def wrap_around(x):
    # x \in [-pi,pi)
    if x >= np.pi:
        return x - 2*np.pi
    elif x < -np.pi:
        return x + 2*np.pi
    else:
        return x

def vectorized_wrap_around(x):
    x[x>=np.pi] -= 2*np.pi
    x[x<-np.pi] += 2*np.pi
    return x

def global_relative_measure(x_target, x_main):
    diff = x_target - x_main
    r = np.sqrt(np.sum(diff**2, axis=1))
    alpha = vectorized_wrap_around(np.arctan2(diff[:,1],diff[:,0]))
    #Rearrange to [r,alpha]*(nb_agents,nb_targets)
    global_state = np.ndarray.flatten(np.vstack((r,alpha)).T)
    return global_state

def cartesian2polar(xy):
    r = np.sqrt(np.sum(xy**2))
    alpha = np.arctan2(xy[1], xy[0])
    return r, alpha

def cartesian2polar_dot(x, y, x_dot, y_dot):
    r2 = x*x + y*y
    if r2 == 0.0:
        return 0.0, 0.0
    r_dot = (x*x_dot + y*y_dot)/np.sqrt(r2)
    alpha_dot = (x*y_dot - x_dot*y)/r2
    return r_dot, alpha_dot

def transform_2d(vec, theta_base, xy_base = [0.0, 0.0]):
    """
    Both vec and frame_xy are in the global coordinate. vec is a vector
    you want to transform with respect to a certain frame which is located at
    frame_xy with ang.
    R^T * (vec - frame_xy).
    R is a rotation matrix of the frame w.r.t the global frame.
    """
    assert(len(vec) == 2)
    return np.matmul([[np.cos(theta_base), np.sin(theta_base)], [-np.sin(theta_base), np.cos(theta_base)]],
        vec - np.array(xy_base))

def rotation_2d_dot(xy_target, xy_dot_target, theta_base, theta_dot_base):
    """
    Cartesian velocity in a rotating frame.
    """
    s_b = np.sin(theta_base)
    c_b = np.cos(theta_base)
    x_dot_target_bframe = (-s_b * xy_target[0] + c_b * xy_target[1]) * theta_dot_base + \
                            c_b * xy_dot_target[0] + s_b * xy_dot_target[1]
    y_dot_target_bframe = - (c_b * xy_target[0] + s_b * xy_target[1]) * theta_dot_base - \
                            s_b * xy_dot_target[0] +c_b * xy_dot_target[1]
    return x_dot_target_bframe, y_dot_target_bframe

def transform_2d_dot(xy_target, xy_dot_target, theta_base, theta_dot_base, xy_base, xy_dot_base):
    """
    Cartesian velocity in a rotating and translating frame.
    """
    rotated_xy_dot_target_bframe = rotation_2d_dot(xy_target, xy_dot_target, theta_base, theta_dot_base)
    rotated_xy_dot_base_bframe = rotation_2d_dot(xy_base, xy_dot_base, theta_base, theta_dot_base)
    return np.array(rotated_xy_dot_target_bframe) - np.array(rotated_xy_dot_base_bframe)

def relative_distance_polar(xy_target, xy_base, theta_base):
    xy_target_base = transform_2d(xy_target, theta_base, xy_base)
    return cartesian2polar(xy_target_base)

def relative_velocity_polar(xy_target, xy_dot_target, xy_base, theta_base, v_base, w_base):
    """
    Relative velocity in a given polar coordinate (radial velocity, angular velocity).

    Parameters:
    ---------
    xy_target : xy coordinate of a target in the global frame.
    xy_dot_target : xy velocity of a target in the global frame.
    xy_base : xy coordinate of the origin of a base frame in the global frame.
    theta_base : orientation of a base frame in the global frame.
    v_base : translational velocity of a base frame in the global frame.
    w_base : rotational velocity of a base frame in the global frame.
    """
    xy_dot_base = vw_to_xydot(v_base, w_base, theta_base)
    xy_target_base = transform_2d(xy_target, theta_base, xy_base)
    xy_dot_target_base = transform_2d_dot(xy_target, xy_dot_target, theta_base,
                                                w_base, xy_base, xy_dot_base)
    r_dot_b, alpha_dot_b = cartesian2polar_dot(xy_target_base[0],
                            xy_target_base[1], xy_dot_target_base[0], xy_dot_target_base[1])
    return r_dot_b, alpha_dot_b

def relative_velocity_polar_se2(xyth_target, vw_target, xyth_base, vw_base):
    """
    Radial and angular velocity of the target with respect to the base frame
    located at xy_b with a rotation of theta_b, moving with a translational
    velocity v and rotational velocity w. This function is designed specifically
    for the SE2 Agent and SE2 Target target case.

    Parameters
    ---------
    xyth_target : (x, y, orientation) of a target in the global frame.
    vw_target : translational and rotational velocity of a target in the global frame.
    xyth_base : (x, y, orientation) of a base frame in the global frame.
    vw_base : translational and rotational velocity of a base frame in the global frame.
    """
    xy_dot_target = vw_to_xydot(vw_target[0], vw_target[1], xyth_base[2])
    return relative_velocity_polar(xyth_target[:2], xy_dot_target, xyth_base[:2],
                                        xyth_base[2], vw_base[0], vw_base[1])

def vw_to_xydot(v, w, theta):
    """
    Conversion from translational and rotational velocity to cartesian velocity
    in the global frame according to differential-drive dynamics.

    Parameters
    ---------
    v : translational velocity.
    w : rotational velocity.
    theta : orientation of a base object in the global frame.
    """
    if w < 0.001:
        x_dot = v * np.cos(theta + w/2)
        y_dot = v * np.sin(theta + w/2)
    else:
        x_dot = v/w * (np.sin(theta + w) - np.sin(theta))
        y_dot = v/w * (np.cos(theta) - np.cos(theta + w))
    return x_dot, y_dot
