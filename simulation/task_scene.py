"""Derived red-brick/grey-bin diagnostic scene; never a hardware calibration.

The arm assets remain the frozen Seeed model. Task dimensions, placements and
the overhead camera are a reconstruction of the public episode, not published
measurements. This module never attaches the brick to the robot or moves it
after MuJoCo initializes its free joint.
"""

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'local/mujoco_upstream/rebotarm_mujoco_rs/models'
SURFACE_HEIGHT_M = 0.0
BRICK_CENTRE_M = (0.340, -0.025, 0.0105)
BRICK_DIMENSIONS_M = (0.032, 0.024, 0.020)
BIN_CENTRE_M = (0.435, 0.180, 0.0)
BIN_OUTSIDE_M = (0.120, 0.120, 0.050)
BIN_WALL_M = 0.004
BIN_YAW_RAD = math.radians(12)


def _numbers(values):
    return ' '.join(f'{value:.9g}' for value in values)


def _grid_texture(path):
    """A visual cutting-mat grid, independent of collision geometry."""
    image = Image.new('RGB', (1024, 1024), (39, 78, 111))
    draw = ImageDraw.Draw(image)
    for coordinate in range(0, 1024, 16):
        bold = coordinate % 128 == 0
        colour = (153, 177, 191) if bold else (82, 119, 146)
        width = 2 if bold else 1
        draw.line((coordinate, 0, coordinate, 1023), fill=colour, width=width)
        draw.line((0, coordinate, 1023, coordinate), fill=colour, width=width)
    draw.line((0, 1023, 1023, 0), fill=(148, 174, 190), width=2)
    for x in range(80, 1024, 140):
        draw.ellipse((x - 20, 810, x + 20, 850), outline=(153, 177, 191), width=2)
    image.save(path)


def build_scene(output_dir, *, square_brick=False, square_edge_m=.032, wrist_camera=None,
                flat_brick=False):
    """Write a standalone derived XML and texture; return (path, metadata).

    The output must be on the same Windows volume as the frozen assets so mesh
    paths can stay relative and pass MuJoCo's native filename API. No upstream
    file is edited. Body/geom names are stable for rollout metrics.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if square_edge_m not in (.032, .040, .048):
        raise ValueError('Supported reconstruction edges are 32, 40 or 48 mm')
    if flat_brick and (not square_brick or square_edge_m != .040):
        raise ValueError('The fixed flat project prop is a 40 x 40 x 20 mm square block')
    dimensions = (square_edge_m, square_edge_m, 0.020) if square_brick else BRICK_DIMENSIONS_M
    stud_columns = 4 if square_brick else 3
    stud_pitch = square_edge_m / 4 if square_brick else .008
    mass = .020 * (square_edge_m / .032) ** 2 if square_brick else .020
    arm_path = UPSTREAM / 'rs_arm.xml'
    original_scene_path = UPSTREAM / 'rs_grasp_scene.xml'
    root = ET.parse(arm_path).getroot()
    original_scene = ET.parse(original_scene_path).getroot()
    root.set('model', 'rebot_rs_red_brick_grey_bin_reconstruction')
    root.find('compiler').set(
        'meshdir', os.path.relpath(UPSTREAM / 'meshes', output_dir).replace('\\', '/'))
    root.insert(1, ET.Element('option', timestep='0.001', iterations='100',
                              solver='Newton', integrator='Euler'))
    for tag in ('statistic', 'visual'):
        root.append(deepcopy(original_scene.find(tag)))
    root.find('statistic').set('center', '0.26 0.04 0.16')
    root.find('statistic').set('extent', '0.8')
    visual = root.find('visual')
    visual.find('headlight').set('ambient', '0.12 0.12 0.12')
    visual.find('headlight').set('diffuse', '0.28 0.28 0.28')
    ET.SubElement(visual, 'map', znear='0.005', zfar='10')
    # Standard 640x480 render targets are large enough for the ACT input images.
    ET.SubElement(visual, 'quality', shadowsize='2048')
    ET.SubElement(visual, 'rgba', fog='0.74 0.73 0.68 1') if visual.find('rgba') is None else None

    asset = root.find('asset')
    texture_path = output_dir / 'task_mat_grid.png'
    _grid_texture(texture_path)
    ET.SubElement(asset, 'texture', name='task_mat_grid_tex', type='2d',
                  file=texture_path.name)
    ET.SubElement(asset, 'material', name='task_mat_blue',
                  texture='task_mat_grid_tex', texuniform='false', texrepeat='1 1',
                  rgba='1 1 1 1', specular='0.06', shininess='0.05')
    ET.SubElement(asset, 'material', name='task_wood', rgba='0.59 0.45 0.30 1',
                  specular='0.10', shininess='0.08')
    ET.SubElement(asset, 'material', name='task_brick_red', rgba='0.80 0.025 0.025 1',
                  specular='0.28', shininess='0.30')
    ET.SubElement(asset, 'material', name='task_bin_grey', rgba='0.46 0.47 0.47 1',
                  specular='0.12', shininess='0.08')

    world = root.find('worldbody')
    ET.SubElement(world, 'light', name='task_key_light', pos='0.3 -0.3 1.3',
                  dir='0.05 0.1 -1', directional='true', diffuse='0.48 0.48 0.48',
                  ambient='0.12 0.12 0.12')
    # The cached upstream camera sits behind its own opaque lens geometry.
    # Move only the derived optical origin ~9.4 mm forward, past the lens.
    # Keep the attached camera orientation/FOV; this is not a measured D405 pose.
    root.find(".//camera[@name='wrist_rgb']").set('pos', '-0.064 0 0.0577')
    if wrist_camera is not None:
        camera = root.find(".//camera[@name='wrist_rgb']")
        camera.set('pos', _numbers(wrist_camera['fitted_camera_local_position_m']))
        camera.set('xyaxes', _numbers(wrist_camera['xml_xyaxes']))
        camera.set('fovy', str(wrist_camera['fovy_fixed_deg']))
        # The inferred optical origin lies inside the approximate upstream
        # camera housing. Its own visual shell must not occlude its sensor.
        # Keep the housing in top/overview views; wrist rendering hides group 4.
        for name in ('d405_wrist_mount', 'd405_camera_body', 'd405_camera_front',
                     'd405_left_imager', 'd405_right_imager'):
            root.find(f".//geom[@name='{name}']").set('group', '4')
    ET.SubElement(world, 'camera', name='overhead_rgb', mode='fixed',
                  pos='0.296 -0.095 0.66', xyaxes='0 -1 0 1 0 0', fovy='55')
    table = ET.SubElement(world, 'body', name='task_table', pos='0.22 0 -0.02')
    ET.SubElement(table, 'geom', name='task_table_geom', type='box',
                  size='0.68 0.62 0.02', material='task_wood',
                  friction='1.0 0.01 0.001', solref='0.003 1',
                  solimp='0.98 0.995 0.0005')
    # The mat is visual paint on the table, rather than a second collision plane.
    ET.SubElement(world, 'geom', name='task_mat', type='box',
                  pos='0.28 0.04 0.00004', size='0.31 0.30 0.00003',
                  material='task_mat_blue', contype='0', conaffinity='0')

    brick = ET.SubElement(world, 'body', name='target_brick',
                          pos=_numbers(BRICK_CENTRE_M), euler='0 0 0.34906585')
    ET.SubElement(brick, 'freejoint', name='target_brick_freejoint')
    ET.SubElement(brick, 'geom', name='target_brick_geom', type='box',
                  size=_numbers(value / 2 for value in dimensions),
                  material='task_brick_red', mass=str(mass if flat_brick else mass - 4 * stud_columns * 0.0001),
                  friction='1.0 0.02 0.002', condim='4',
                  solref='0.003 1', solimp='0.98 0.995 0.0005')
    # Decoded wrist frames 170/190/200 show four by four studs. The old 4x3
    # scene remains reproducible; millimetre dimensions and mass are unmeasured.
    for row in range(0 if flat_brick else 4):
        for column in range(stud_columns):
            ET.SubElement(brick, 'geom', name=f'target_brick_stud_{row}_{column}',
                          type='cylinder', size=_numbers((stud_pitch * .3125, .0009)),
                          pos=_numbers((-1.5 * stud_pitch + row * stud_pitch,
                                        -(stud_columns - 1) * stud_pitch / 2 + column * stud_pitch,
                                        dimensions[2] / 2 + .0009)),
                          material='task_brick_red', mass='0.0001',
                          friction='1.0 0.02 0.002', solref='0.003 1')

    bin_body = ET.SubElement(world, 'body', name='target_bin',
                            pos=_numbers(BIN_CENTRE_M),
                            euler=_numbers((0, 0, BIN_YAW_RAD)))
    half_x, half_y, height = (BIN_OUTSIDE_M[0] / 2,
                              BIN_OUTSIDE_M[1] / 2, BIN_OUTSIDE_M[2])
    floor_half = BIN_WALL_M / 2
    wall_options = dict(type='box', material='task_bin_grey',
                        friction='0.8 0.01 0.001', solref='0.003 1',
                        solimp='0.98 0.995 0.0005')
    ET.SubElement(bin_body, 'geom', name='target_bin_floor',
                  pos=_numbers((0, 0, floor_half)),
                  size=_numbers((half_x, half_y, floor_half)), **wall_options)
    for name, position, size in (
        ('n', (0, half_y - floor_half, height / 2), (half_x, floor_half, height / 2)),
        ('s', (0, -half_y + floor_half, height / 2), (half_x, floor_half, height / 2)),
        ('e', (half_x - floor_half, 0, height / 2), (floor_half, half_y, height / 2)),
        ('w', (-half_x + floor_half, 0, height / 2), (floor_half, half_y, height / 2)),
    ):
        ET.SubElement(bin_body, 'geom', name=f'target_bin_wall_{name}',
                      pos=_numbers(position), size=_numbers(size), **wall_options)

    # Retain all arm exclusions, including the generic scene's adjacent links.
    contact = root.find('contact')
    pairs = {(item.get('body1'), item.get('body2')) for item in contact}
    for item in original_scene.find('contact'):
        if (item.get('body1'), item.get('body2')) not in pairs:
            contact.append(deepcopy(item))

    scene_path = output_dir / 'task_scene.xml'
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(scene_path, encoding='utf-8', xml_declaration=True)
    metadata = {
        'scene_id': ('fixed_flat40_red_block_grey_bin_v1' if flat_brick else
                     'red_brick_grey_bin_reconstruction_v2_4x4' if square_brick
                     else 'red_brick_grey_bin_reconstruction_v1'),
        'upstream_arm_commit': '438f7713d83fb8cd0d0cb05e8cc335264dd3997a',
        'upstream_arm_sha256': hashlib.sha256(arm_path.read_bytes()).hexdigest(),
        'upstream_files_modified': False,
        'official_task_dimensions_verified': False,
        'scene_calibration_verified': False,
        'placement_basis': 'Approximate public episode 0 object appearance and candidate-mapped grasp/release TCP locations. No policy outcome was used to choose placement.',
        'work_surface_height_m': SURFACE_HEIGHT_M,
        'mat_dimensions_m': [0.62, 0.60],
        'brick': {'body': 'target_brick', 'freejoint': 'target_brick_freejoint',
                  'main_geom': 'target_brick_geom', 'centre_m': list(BRICK_CENTRE_M),
                  'body_dimensions_m': list(dimensions),
                  'stud_layout': [0, 0] if flat_brick else [4, stud_columns],
                  'flat_top': flat_brick,
                  'layout_basis': ('User-defined flat square prop matching cad/pick_place_props_20261023, not an official measured object'
                                   if flat_brick else
                                   '16 studs counted in original wrist frames 170/190/200; exact physical dimensions unverified'
                                   if square_brick else 'Initial 4x3 approximation, superseded by decoded source frames'),
                  'stud_height_m': 0.0 if flat_brick else 0.0018, 'yaw_rad': math.radians(20),
                  'mass_kg': mass, 'mass_and_dimensions': 'illustrative, not measured',
                  'enlarged_size_comparison': bool(not flat_brick and square_brick and square_edge_m != .032)},
        'bin': {'body': 'target_bin', 'centre_m': list(BIN_CENTRE_M),
                'outside_dimensions_m': list(BIN_OUTSIDE_M),
                'inner_dimensions_m': [0.112, 0.112],
                'wall_thickness_m': BIN_WALL_M, 'floor_thickness_m': BIN_WALL_M,
                'yaw_rad': BIN_YAW_RAD, 'construction': 'static floor plus four solid walls; open top'},
        'cameras': {
            'top': {'name': 'overhead_rgb', 'resolution': [640, 480],
                    'pos_m': [0.296, -0.095, 0.66], 'fovy_deg': 55,
                    'calibration': 'rough reconstruction, not camera extrinsics from the author'},
            'wrist': {'name': 'wrist_rgb', 'resolution': [640, 480],
                      'optical_origin_local_m': [-0.064, 0, 0.0577],
                      'optical_origin_adjustment': 'Derived camera moved approximately 9.4 mm forward of upstream origin to avoid opaque self-lens occlusion; orientation and FOV unchanged.',
                      'calibration': 'upstream mount/orientation with corrected visible origin; not verified against author camera mounting'},
        },
        'dynamic_brick': True,
        'grasp_weld_or_attachment': False,
        'post_initialization_object_teleportation': False,
        'limitations': [
            'Appearance, camera extrinsics, toy dimensions, lighting, friction and zero alignment differ from the real data.',
            'Removing the generic 0.10 m-high table corrects a scene incompatibility, not a verified robot calibration.',
            'This reconstructed simulator is a diagnostic transfer test, not the collection environment or evidence of real-robot success.',
        ],
    }
    if wrist_camera is not None:
        metadata['cameras']['wrist'] = {
            'name': 'wrist_rgb', 'resolution': [640, 480],
            'calibration': 'Inferred from real demonstration centroids and candidate FK, not hardware hand-eye calibration',
            'own_camera_visual_shell_group': 4,
            'wrist_render_excludes_own_camera_shell': True,
            'fit': deepcopy(wrist_camera),
        }
    return scene_path, metadata
