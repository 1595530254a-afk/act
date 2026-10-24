"""Candidate ACT-coordinate mappings for diagnostic simulation, never hardware.

The dataset's collection-time plugin/configuration/zero convention is unknown.
No limit clipping, sign fitting, offset fitting, or physical validation occurs
here. A finite output is a conversion result, not a valid-pose or grasp pass.
"""
from copy import deepcopy

import numpy as np


MODEL_CHANNELS = (
    'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
    'wrist_flex.pos', 'wrist_yaw.pos', 'wrist_roll.pos', 'gripper.pos',
)
SIM_JOINT_NAMES = tuple(f'joint{index}' for index in range(1, 8))
LEGACY_PLUGIN105_ACTION_FACTORS = (1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 6.0)
MAPPING_STATUS = 'CANDIDATE_UNVERIFIED'
# Native LeRobot's public state/action frame is shared. Its RS profile maps
# every public joint degree to the opposite native motor degree; no gripper x6.
DIRECTIONS = (-1.0,) * 7
GRIPPER_MOTOR_OPEN_RAD = 5.0
# Use the source controller's nominal 90 mm total opening at 5 motor radians.
# The old display bridge expanded 45 mm to 50 mm per finger; applying that
# expansion to the physical collision model made the grasp 11.1% too open.
SIM_GRIPPER_OPEN_M = 0.045
GRIPPER_METRES_PER_MOTOR_RAD = SIM_GRIPPER_OPEN_M / GRIPPER_MOTOR_OPEN_RAD
LEGACY_BRIDGE_METRES_PER_MOTOR_RAD = 0.01

_METADATA = {
    'mapping_id': 'candidate_lerobot_rs_nominal_90mm_gripper_v3',
    'status': MAPPING_STATUS,
    'hardware_io': False,
    'hardware_execution_permitted': False,
    'dataset_collection_config_verified': False,
    'dataset_units_explicit_in_meta': False,
    'physical_zero_alignment_verified': False,
    'signs_or_offsets_fitted_to_dataset': False,
    'silent_limit_clipping': False,
    'model_channels': list(MODEL_CHANNELS),
    'output_channels': list(SIM_JOINT_NAMES),
    'output_units': ['rad'] * 6 + ['m'],
    'state_mapping': 'candidate canonical public degrees * RS directions [-1]*7 -> native radians; joint7 radians * 0.045/5 metres',
    'gripper_metres_per_motor_rad': GRIPPER_METRES_PER_MOTOR_RAD,
    'gripper_nominal_total_opening_m': 0.09,
    'previous_display_bridge_ratio': LEGACY_BRIDGE_METRES_PER_MOTOR_RAD,
    'gripper_physical_measurement_verified': False,
    'action_mapping': 'same canonical public coordinate frame as state, BEFORE any clipping -> same RS simulation conversion',
    'state_action_share_public_coordinate_frame': True,
    'canonical_public_to_rs_native_factors': list(DIRECTIONS),
    'legacy_plugin105_action_factors_applied_to_state': False,
    'legacy_plugin105_action_factors': list(LEGACY_PLUGIN105_ACTION_FACTORS),
    'legacy_diagnostic_functions': ['legacy_plugin105_state_to_sim', 'legacy_plugin105_action_to_sim'],
    'gripper_finger_qpos': 'joint_left = joint_right = joint7; XML joint axes encode the mirror',
    'limitations': [
        'Dataset metadata supplies channel order but not collection-time physical units or zero calibration.',
        'Native LeRobot rebot_b601_follower and external seeed_b601 plugin1.0.5 expose different coordinate conventions; robot_type alone does not make them interchangeable.',
        'The publisher collection CLI, motor-family revision and zero calibration are not available; canonical mapping is evidence-based but remains a transfer candidate.',
        'No signs/offsets are fitted merely to make values fall inside the current model limits; incompatible values remain visible.',
        'Canonical action outputs deliberately omit public-limit/relative clipping for diagnostics.',
        'The 0.009 m/rad gripper conversion uses source nominal width/endpoints, not an independent physical opening or force measurement.',
    ],
    'sources': [
        {'kind': 'native_lerobot_motor_family_profile',
         'url': 'https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/robots/rebot_b601_follower/motor_family.py#L119-L167',
         'evidence': 'RS public state/action directions are -1 for all seven joints, including gripper; no factor 6.'},
        {'kind': 'native_lerobot_public_coordinate_contract',
         'url': 'https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L244-L316',
         'evidence': 'get_observation converts motor degrees to public degrees by division; action uses the same public degrees and multiplies direction before native radians.'},
        {'kind': 'local_dataset_metadata', 'path': 'data/rebot_cube/meta/info.json',
         'sha256': 'dbf3d191a8e383111a26f5b2c8626bde59abf18a083fb8ba044419efb5d39ac5',
         'evidence': 'Identical seven-channel state/action order; no declared physical unit.'},
        {'kind': 'local_laptop_plugin_1.0.5',
         'path': 'local/executor_return01_20261021_195243/context/vendor_sources/lerobot_robot_seeed_b601_1.0.5/lerobot_robot_seeed_b601/seeed_b601_follower.py',
         'sha256': '5910ad62c518308c20c64ec851a7c5389eff0ca7f4cf9b493696ab3addd63fd7',
         'evidence': 'External plugin convention, retained only for explicitly named legacy diagnostics: get_observation L597-634 reads motor radians as degrees without action factors; send_action L636-699 multiplies factors then clips.'},
        {'kind': 'local_laptop_plugin_config_1.0.5',
         'path': 'local/executor_return01_20261021_195243/context/vendor_sources/lerobot_robot_seeed_b601_1.0.5/lerobot_robot_seeed_b601/config_seeed_b601_rs_follower.py',
         'sha256': 'ae0539a7cb113b8532679620079e3dfaf52e354f40eb3ad1e6e681a8ec9487c8',
         'evidence': 'CAN IDs 1..7 and default action factors [1,1,-1,-1,-1,1,6].'},
        {'kind': 'wiki_linked_author_controller',
         'url': 'https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_RS/blob/aa807cf2affe85b13b86221159871aae7742c01b/rebotarm_ros2_RS/src/rebotarmcontroller/rebotarmcontroller/ros_publishers.py#L85-L151',
         'evidence': 'Publishes six hardware joint radians directly; converts gripper open 5 rad to ROS single-finger 0.045 m.'},
        {'kind': 'official_sim_bridge',
         'url': 'https://github.com/Seeed-Projects/reBot_RS_Sim/blob/438f7713d83fb8cd0d0cb05e8cc335264dd3997a/rebotarm_mujoco_rs/rebotarm_mujoco_rs/mujoco_sync.py',
         'evidence': 'L627-639 arm radians copied directly; L758-761 expands ROS finger 0.045 m to display joint7 0.05 m. That display expansion is deliberately omitted from the v3 physical collision experiment.'},
        {'kind': 'official_sim_driver',
         'url': 'https://github.com/Seeed-Projects/reBot_RS_Sim/blob/438f7713d83fb8cd0d0cb05e8cc335264dd3997a/rebotarm_mujoco_rs/rebotarm_mujoco_rs/rs_driver.py#L19-L29',
         'evidence': 'Current model has J2/J3 native joint limits 0..3.14 rad and gripper motor-open convention 5 rad.'},
    ],
}


def mapping_metadata():
    """Return independent evidence metadata for saved diagnostic reports."""
    return deepcopy(_METADATA)


def _seven_real(values):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError('Expected a real numeric array with shape (..., 7)') from error
    if array.ndim == 0 or array.shape[-1] != 7 or array.dtype.kind not in 'fiu':
        raise ValueError('Expected a real numeric array with shape (..., 7); bool/text/complex/object are rejected')
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError('Every coordinate must be finite')
    return array


def _native_degrees_to_sim(native_degrees, gripper_ratio=GRIPPER_METRES_PER_MOTOR_RAD):
    # Six native arm angles stay direct. For gripper contacts use the source
    # nominal physical width, rather than the bridge's expanded display width.
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.deg2rad(native_degrees)
        result[..., 6] *= gripper_ratio
    if not np.isfinite(result).all():
        raise ValueError('Coordinate conversion overflowed; no target returned')
    return result


def dataset_state_to_sim(values):
    """Candidate native-LeRobot PUBLIC observations -> six RS rad + slide m.

    Apply the documented canonical-public-to-RS-native sign to all seven
    channels. Do not apply external plugin1.0.5's asymmetric action factors.
    The caller must report limits against its loaded model; no clipping occurs.
    """
    return _native_degrees_to_sim(_seven_real(values) * np.asarray(DIRECTIONS))


def model_action_to_sim(values):
    """Candidate native-LeRobot PUBLIC PRECLIP actions -> RS simulation.

    State and action share the public frame in native LeRobot, so conversion
    is identical. The collection configuration is still unverified. This does
    not reproduce downstream clipping or permit hardware execution.
    """
    return _native_degrees_to_sim(_seven_real(values) * np.asarray(DIRECTIONS))


def legacy_plugin105_state_to_sim(values):
    """Explicit old external-plugin hypothesis; never the default dataset map."""
    return _native_degrees_to_sim(_seven_real(values), LEGACY_BRIDGE_METRES_PER_MOTOR_RAD)


def legacy_plugin105_action_to_sim(values):
    """Explicit external-plugin1.0.5 PRECLIP diagnostic; not canonical LeRobot."""
    action = _seven_real(values)
    with np.errstate(over='ignore', invalid='ignore'):
        native_degrees = action * np.asarray(LEGACY_PLUGIN105_ACTION_FACTORS)
    return _native_degrees_to_sim(native_degrees, LEGACY_BRIDGE_METRES_PER_MOTOR_RAD)
