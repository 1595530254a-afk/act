"""Simulation-only torque servo for ACT closed-loop experiments.

These gains are numerical experiment settings, not B601 motor parameters.  The
servo only applies forces: it never assigns joint/object positions after the
explicit initial-pose operation and never attaches an object to the gripper.
"""

import mujoco
import numpy as np

from joint_mapping import DIRECTIONS, GRIPPER_METRES_PER_MOTOR_RAD, SIM_JOINT_NAMES


# Proportional gains: N m/rad for the six hinges, N/m for the slide.
# Derivative gains: N m s/rad and N s/m respectively.  Both are sim-only.
KP = np.array([120.0, 160.0, 120.0, 25.0, 20.0, 10.0, 3500.0])
KD = np.array([6.0, 8.0, 5.0, 0.8, 0.6, 0.3, 24.0])
FINGER_JOINTS = ('joint_left', 'joint_right')


def _pose(values):
    array = np.asarray(values)
    if array.shape != (7,) or array.dtype.kind not in 'fiu':
        raise ValueError('Expected seven real native simulator coordinates')
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError('Simulator coordinates must be finite')
    return array


def _addresses(model, names, attribute):
    return np.array([int(getattr(model.joint(name), attribute)[0]) for name in names])


def validate_target(model, values):
    """Return a copy of an in-range target; do not silently clip a policy."""
    target = _pose(values)
    for index, name in enumerate(SIM_JOINT_NAMES):
        joint = model.joint(name)
        if bool(joint.limited[0]):
            lower, upper = joint.range
            if not lower <= target[index] <= upper:
                raise ValueError(f'{name} target {target[index]:.9g} outside [{lower}, {upper}]')
    return target


def initialize(model, data, pose_native):
    """Set robot initial coordinates once, preserving free object state/time.

    The caller owns resetting/creating MjData for each rollout.  Only robot
    positions, robot velocities and the seven actuator commands are changed.
    """
    pose = validate_target(model, pose_native)
    names = (*SIM_JOINT_NAMES, *FINGER_JOINTS)
    addresses = _addresses(model, names, 'qposadr')
    data.qpos[addresses] = np.r_[pose, pose[6], pose[6]]
    data.qvel[_addresses(model, names, 'dofadr')] = 0.0
    for name in SIM_JOINT_NAMES:
        data.ctrl[model.actuator(f'{name}_motor').id] = 0.0
    mujoco.mj_forward(model, data)


def read_native_state(model, data):
    """Read six hinge radians and the master gripper slide in metres."""
    return data.qpos[_addresses(model, SIM_JOINT_NAMES, 'qposadr')].copy()


def native_to_public(values):
    """Inverse of the existing candidate canonical public-to-sim mapping."""
    native = _pose(values)
    native[6] /= GRIPPER_METRES_PER_MOTOR_RAD
    return np.rad2deg(native) / np.asarray(DIRECTIONS)


def read_public_state(model, data):
    """ACT observation in the candidate publisher/public degree convention."""
    return native_to_public(read_native_state(model, data))


class ServoController:
    """PD plus model bias forces, with the XML actuator force limits.

    apply() writes data.ctrl and does not advance time; step() additionally
    executes one MuJoCo step.  There is no hardware API in this module.
    """

    def __init__(self, model):
        self.model = model
        self.qpos_addresses = _addresses(model, SIM_JOINT_NAMES, 'qposadr')
        self.dof_addresses = _addresses(model, SIM_JOINT_NAMES, 'dofadr')
        self.finger_dofs = _addresses(model, FINGER_JOINTS, 'dofadr')
        self.actuator_ids = np.array([model.actuator(f'{name}_motor').id for name in SIM_JOINT_NAMES])
        self.limits = model.actuator_ctrlrange[self.actuator_ids].copy()
        # The named XML actuators are direct unit-gear motors, so ctrl is the
        # joint torque/force. Reject an incompatible model rather than assume.
        for actuator_id, dof_id in zip(self.actuator_ids, self.dof_addresses):
            joint_id = int(model.dof_jntid[dof_id])
            if (model.actuator_trnid[actuator_id, 0] != joint_id
                    or not np.allclose(model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0])
                    or not model.actuator_ctrllimited[actuator_id]):
                raise ValueError('Expected named, limited, direct unit-gear joint motors')
        self.saturated_steps = np.zeros(7, dtype=np.int64)
        self.steps_applied = 0

    def apply(self, data, target_native):
        target = validate_target(self.model, target_native)
        position = data.qpos[self.qpos_addresses]
        velocity = data.qvel[self.dof_addresses]
        bias = data.qfrc_bias[self.dof_addresses].copy()
        # Equality couples all three finger coordinates 1:1.  A single master
        # actuator must support the bias of both passive finger slides too.
        bias[6] += data.qfrc_bias[self.finger_dofs].sum()
        requested = KP * (target - position) - KD * velocity + bias
        if not np.isfinite(requested).all():
            raise FloatingPointError('Non-finite simulator servo force')
        limited = np.clip(requested, self.limits[:, 0], self.limits[:, 1])
        self.saturated_steps += (limited != requested)
        self.steps_applied += 1
        data.ctrl[self.actuator_ids] = limited
        return limited.copy()

    def step(self, data, target_native):
        control = self.apply(data, target_native)
        mujoco.mj_step(self.model, data)
        return control

    def metadata(self):
        return {
            'controller': 'simulation_only_PD_plus_MuJoCo_bias',
            'kp': KP.tolist(), 'kd': KD.tolist(),
            'units': ['N_m_per_rad'] * 6 + ['N_per_m'],
            'xml_force_limits': self.limits.tolist(),
            'steps_applied': self.steps_applied,
            'saturated_steps_per_joint': self.saturated_steps.tolist(),
            'targets_clipped': False, 'positions_assigned_after_initialization': False,
            'object_attachment_or_teleport': False, 'hardware_io': False,
            'mapping_status': 'CANDIDATE_UNVERIFIED',
        }
