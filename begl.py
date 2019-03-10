"""
BEGL Toolbar - A Blender toolbar for creating and exporting bgl format.

BLENDER consists of a single SCENE

    SCENE - collection of MODEL
    MODEL - collection of MPART
    MPART - BLENDER OBJECT
                                    
    Documentation to be added

                                        # export format
SCENE       
LM          name l data
LRSM        name l r s data
ANIMATION   name fps keyframe_count
KEYFRAME    keyframe_index l r s data
DIFFUSEMAP  name relative_path 

MODEL name  
LM          name l data
LRSM        name l r s, data
ANIMATION   name fps keyframe_count
KEYFRAME    keyframe_index l r s data
DIFFUSEMAP  name relative_path

MPART name  
LM          name l data
LRSM        name l r s, data
ANIMATION   name fps keyframe_count
KEYFRAME    keyframe_index l r s data
DIFFUSEMAP  name relative_path
V  vertex
VT vertex Texture Coordinate
VN VERTEX Normal
F v/vt/vn
"""

import bpy
import math
import bmesh
import os

# ACTION TYPE 
ACTION_TYPE_SCENE         = 'SCENE_ACTION'
ACTION_TYPE_MODEL         = 'MODEL_ACTION'
ACTION_TYPE_MPART         = 'MPART_ACTION'

#ACTION_KIND
ACTION_KIND_LM            = 'LM'
ACTION_KIND_LRSM          = 'LRSM'
ACTION_KIND_ANIMATION     = 'ANIMATION'

# IMAGE TYPE
IMAGE_TYPE_SCENE_DIFFUSE      = 'SCENE_DIFFUSE'
IMAGE_TYPE_MODEL_DIFFUSE      = 'MODEL_DIFFUSE'
IMAGE_TYPE_MPART_DIFFUSE      = "MPART_DIFFUSE"

# OBJECT TYPE
BLENDER_OBJECT                = 'BLENDER'
EXPORT_OBJECT                 = 'EXPORT'
MODEL_OBJECT                  = 'MO'
SCENE_OBJECT                = 'SO'

scene = bpy.context.scene

def get_frame_numbers(obj, action):
    if not obj.animation_data:
        obj.animation_data_create()

    obj.animation_data.action = action

    frames = set()
    anim = obj.animation_data
    for fcu in anim.action.fcurves:
        for keyframe in fcu.keyframe_points:
            x = math.ceil(keyframe.co.x)
            frames.add(x)
    return sorted(frames)

def vec3_to_str(vec):
    return '%.6f %.6f %.6f' % (vec.x, vec.y, vec.z)

def vec2_to_str(vec):
    return '%.6f %.6f' % (vec.x, vec.y)

def quat_to_str(quat):
    return '%.6f %.6f %.6f %.6f' % (quat.w, quat.x, quat.y, quat.z)

class DiffuseMap:
    def __init__(self, base_path, image):
        self.base_path = base_path
        self.image = image 
        self.name  = '-'.join(image.name.split('-')[1:])

    def __str__(self):
        self.save_image()
        return "DIFFUSEMAP %s %s" % (self.name, self.name + '.png')

    def save_image(self):
        self.image.filepath_raw = os.path.join(self.base_path, self.name + '.png')
        self.image.file_format = 'PNG'
        self.image.save()

class KeyFrame:

    def __init__(self, obj, action, frame_number):
        self.obj = obj
        self.action = action
        self.frame_number = frame_number

        if not self.obj.animation_data:
            self.obj.animation_data_create()

        self.obj.animation_data.action = self.action
        scene.frame_set(self.frame_number)

        loc_, rot_, scale_ = self.obj.matrix_world.decompose()
        self.loc, self.rot, self.scale = (vec3_to_str(loc_), quat_to_str(rot_), vec3_to_str(scale_))

    def decompose(self):
        return (self.loc, self.rot, self.scale)

class Lm:
    def __init__(self, obj, action):
        self.obj = obj
        self.action = action
        self.name   = action.name.split('-')[2]

    def __str__(self):
        key = KeyFrame(self.obj, self.action, 0)
        return '%s %s ' % (ACTION_KIND_LM, self.name) + key.loc

class Lrsm:
    def __init__(self, obj, action):
        self.obj = obj
        self.action = action
        self.name   = action.name.split('-')[2]

    def __str__(self):
        out = ' '.join(KeyFrame(self.obj, self.action, 0).decompose())
        return '%s %s ' % (ACTION_KIND_LRSM, self.name) + out

class Animation:
    def __init__(self, obj, action):
        self.obj = obj
        self.action = action
        self.name   = action.name.split('-')[2]

    def __str__(self):
        frames = get_frame_numbers(self.obj, self.action)
        out = ['%s %s %d %d' % (ACTION_KIND_ANIMATION, self.name, scene.render.fps, len(frames))]
        for f in frames:
            lrs_str = ' '.join(KeyFrame(self.obj, self.action, f).decompose())
            frame_str = 'KEYFRAME %d ' % (f) + lrs_str
            out.append(frame_str)
        return '\n'.join(out)


def find_images(IMAGE_TYPE, DIFFUSE_NAME = None, MODEL_NAME = None, MPART_NAME = None):
    output = []
    for image in bpy.data.images:
        blocks = image.name.split('-')
        if not IMAGE_TYPE == blocks[0]:
            continue
        if DIFFUSE_NAME and not DIFFUSE_NAME == blocks[1]:
            continue
        if MODEL_NAME and not MODEL_NAME == blocks[2]:
            continue
        if MPART_NAME and not MPART_NAME == blocks[3]:
            continue
        output.append(image)
    return output


def find_action(ACTION_TYPE, ACTION_KIND = None, ACTION_NAME = None, MODEL_NAME = None, MPART_NAME = None):
    output = []
    for action in bpy.data.actions:
        blocks = action.name.split('-')
        if not ACTION_TYPE == blocks[0]:
            continue
        if ACTION_KIND and not ACTION_KIND == blocks[1]:
            continue
        if ACTION_NAME and not ACTION_NAME == blocks[2]:
            continue
        if MODEL_NAME  and not MODEL_NAME == blocks[3]:
            continue
        if MPART_NAME  and not MPART_NAME == blocks[4]:
            continue
        output.append(action)
    return output


def find_meshed_objects(OBJECT_TYPE, MODEL_NAME=None, MPART_NAME=None):

    output = []
    for obj in bpy.data.objects:
        if not obj.type == 'MESH':
            continue
        blocks = obj.name.split('-')
        if not OBJECT_TYPE == blocks[0]:
            continue
        if MODEL_NAME and not MODEL_NAME == blocks[1]:
            continue
        if MPART_NAME and not MPART_NAME == blocks[2]:
            continue
        output.append(obj)
    return output

def find_objects(OBJECT_TYPE, name):
    me = bpy.data.meshes.new("mesh datablock name")
    ob = bpy.data.objects.new("object name", me)
    bpy.context.scene.objects.link(ob)

def get_scene_model_names():
    return set(obj.name.split('-')[1] for obj in find_meshed_objects(MODEL_OBJECT))

def get_model_mpart_names(model_name):
    return set(obj.name.split('-')[2] for obj in find_meshed_objects(EXPORT_OBJECT, model_name))

def generate_mpart_data(base_path, model_name, mpart_name):

    obj = find_meshed_objects(EXPORT_OBJECT, model_name, mpart_name)[0]
    MPART_DATA = {
        'MODEL_NAME'   : model_name, 
        'MPART_NAME'   : mpart_name, 
        'MPART_OBJECT' : obj, 
        'DIFFUSE_MAPS' : [], 

        'ACTION': {
            ACTION_KIND_LM          : [], 
            ACTION_KIND_LRSM        : [], 
            ACTION_KIND_ANIMATION   : [], 
        }
    }

    for image in find_images(IMAGE_TYPE_MPART_DIFFUSE, None, model_name, mpart_name):
        MPART_DATA['DIFFUSE_MAPS'].append(DiffuseMap(base_path, image))

    for m in find_action(ACTION_TYPE_MPART, ACTION_KIND_LM, None, model_name, mpart_name):
        MPART_DATA['ACTION'][ACTION_KIND_LM].append(Lm(obj, m))

    for m in find_action(ACTION_TYPE_MPART, ACTION_KIND_LRSM, None, model_name, mpart_name):
        MPART_DATA['ACTION'][ACTION_KIND_LRSM].append(Lrsm(obj, m))

    for m in find_action(ACTION_TYPE_MPART, ACTION_KIND_ANIMATION, None, model_name, mpart_name):
        MPART_DATA['ACTION'][ACTION_KIND_ANIMATION].append(Animation(obj, m))

    return MPART_DATA

def generate_model_data(base_path, model_name):

    mmo = find_meshed_objects(MODEL_OBJECT, model_name)
    mmo = mmo[0] if mmo else None

    MODEL_DATA = {
        'MODEL_NAME'   : model_name, 
        'DIFFUSE_MAPS' : [], 

        'ACTION': {
            ACTION_KIND_LM          : [], 
            ACTION_KIND_LRSM        : [], 
            ACTION_KIND_ANIMATION   : [], 
        },

        'MPARTS'       : []
    }

    for img in find_images(IMAGE_TYPE_MODEL_DIFFUSE, None, model_name):
        MODEL_DATA['DIFFUSE_MAPS'].append(DiffuseMap(base_path, img))

    for m in find_action(ACTION_TYPE_MODEL, ACTION_KIND_LM, None, model_name):
        MODEL_DATA['ACTION'][ACTION_KIND_LM].append(Lm(mmo, m))

    for m in find_action(ACTION_TYPE_MODEL, ACTION_KIND_LRSM, None, model_name):
        MODEL_DATA['ACTION'][ACTION_KIND_LRSM].append(Lrsm(mmo, m))

    for m in find_action(ACTION_TYPE_MODEL, ACTION_KIND_ANIMATION, None, model_name):
        MODEL_DATA['ACTION'][ACTION_KIND_ANIMATION].append(Animation(mmo, m))

    for mpart_name in get_model_mpart_names(model_name):
        MODEL_DATA['MPARTS'].append(generate_mpart_data(base_path, model_name, mpart_name))

    return MODEL_DATA

def is_action_present(action_name):
    for action in bpy.data.actions:
        if action.name == action_name:
            return True
    return False 

def is_image_present(image_name):
    for image in bpy.data.images:
        if image.name == image_name:
            return True 
    return False

def mesh_triangulate(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()

class Face:
    def __init__(self, polygon_index, polygon):
        self.vertex_indices = [v for v in polygon.vertices]
        self.normal_index   = polygon_index
        self.uv_indices     = [uv for uv in polygon.loop_indices]
    
    def __str__(self):
        ret = ['F']
        for v, vt in zip(self.vertex_indices, self.uv_indices):
            vertex = "%d/%d/%d" % (v, self.normal_index, vt)
            ret.append(vertex)
        return ' '.join(ret)

def generate_mesh_export_str(mesh):
    mesh_triangulate(mesh)

    vertices = [v.co  for v in mesh.vertices]
    tex      = [uv.uv for uv in mesh.uv_layers[0].data]
    normals  = [p.normal for p in mesh.polygons]

    out = []

    for index, polygon in enumerate(mesh.polygons):
        vertex_indices  = [v  for v in polygon.vertices]
        uv_indices = [uv for uv in polygon.loop_indices]
        normal_index = index

        for v, vt in zip(vertex_indices, uv_indices):
            vertex = 'VERTEX %s %s %s' % (vec3_to_str(vertices[v]), vec2_to_str(tex[vt]), vec3_to_str(normals[normal_index]))
            out.append(vertex)
    return (len(out), '\n'.join(out))

def generate_mpart_export_str(MPART_DATA):
    vertex_len, mesh_str = generate_mesh_export_str(MPART_DATA['MPART_OBJECT'].data)

    mpart_header = ['MPART %s %d' % (MPART_DATA['MPART_NAME'], vertex_len)]

    for dm in MPART_DATA['DIFFUSE_MAPS']:
        mpart_header.append(str(dm))

    for lm in MPART_DATA['ACTION'][ACTION_KIND_LM]:
        mpart_header.append(str(lm))

    for lrsm in MPART_DATA['ACTION'][ACTION_KIND_LRSM]:
        mpart_header.append(str(lrsm))

    for anim in MPART_DATA['ACTION'][ACTION_KIND_ANIMATION]:
        mpart_header.append(str(anim))

    mpart_header.append(mesh_str)
    return '\n'.join(mpart_header)


def generate_model_export_str(MODEL_DATA):
    model_header = ['MODEL %s' % (MODEL_DATA['MODEL_NAME'])]

    for dm in MODEL_DATA['DIFFUSE_MAPS']:
        model_header.append(str(dm))

    for lm in MODEL_DATA['ACTION'][ACTION_KIND_LM]:
        model_header.append(str(lm))

    for lrsm in MODEL_DATA['ACTION'][ACTION_KIND_LRSM]:
        model_header.append(str(lrsm))

    for anim in MODEL_DATA['ACTION'][ACTION_KIND_ANIMATION]:
        model_header.append(str(anim))

    for mpart in MODEL_DATA['MPARTS']:
        model_header.append(generate_mpart_export_str(mpart))

    return '\n'.join(model_header)

def generate_scene_export_str(SCENE_DATA, base_path):
    model_data = []

    for name in SCENE_DATA['MODEL_NAMES']:
        model_data.append(generate_model_data(base_path, name))

    scene_header = ['SCENE']
    for dm in SCENE_DATA['DIFFUSE_MAPS']:
        scene_header.append(str(dm))

    for lm in SCENE_DATA['ACTION'][ACTION_KIND_LM]:
        scene_header.append(str(lm))

    for lrsm in SCENE_DATA['ACTION'][ACTION_KIND_LRSM]:
        scene_header.append(str(lrsm))

    for anim in SCENE_DATA['ACTION'][ACTION_KIND_ANIMATION]:
        scene_header.append(str(anim))

    for model in model_data:
        scene_header.append(generate_model_export_str(model))

    return '\n'.join(scene_header) + '\n'

def export_scene(base_path, filename):

    def fill_scene_data(SCENE_DATA, base_path):
        smo = find_meshed_objects(SCENE_OBJECT)
        smo = smo[0] if smo else None

        for m in find_action(ACTION_TYPE_SCENE, ACTION_KIND_LM):
            SCENE_DATA['ACTION'][ACTION_KIND_LM].append(Lm(smo, m))

        for m in find_action(ACTION_TYPE_SCENE, ACTION_KIND_LRSM):
            SCENE_DATA['ACTION'][ACTION_KIND_LRSM].append(Lrsm(smo, m))

        for m in find_action(ACTION_TYPE_SCENE, ACTION_KIND_ANIMATION):
            SCENE_DATA['ACTION'][ACTION_KIND_ANIMATION].append(Animation(smo, m))

        for image in find_images(IMAGE_TYPE_SCENE_DIFFUSE):
            SCENE_DATA['DIFFUSE_MAPS'].append(DiffuseMap(base_path, image))

    SCENE_DATA = {
        'DIFFUSE_MAPS': [], 

        'ACTION': {
            ACTION_KIND_LM          : [], 
            ACTION_KIND_LRSM        : [], 
            ACTION_KIND_ANIMATION   : [], 
        },

        'MODEL_NAMES': get_scene_model_names()
    }

    fill_scene_data(SCENE_DATA, base_path)

    with open(os.path.join(base_path, filename), 'w') as f:
        out = generate_scene_export_str(SCENE_DATA, base_path)
        f.write(out)
    f.close()
    return True

def is_object_present(OBJECT_TYPE, MODEL_NAME):
    return len(find_meshed_objects(OBJECT_TYPE, MODEL_NAME))

def is_mpart_present(OBJECT_TYPE, MODEL_NAME, MPART_NAME):
    return len(find_meshed_objects(OBJECT_TYPE, MODEL_NAME, MPART_NAME))

class BEGLToolbar(bpy.types.Operator):
    bl_idname = 'object.begl_toolbar'
    bl_label  = 'begl toolbar'

    def get_property_values(self):
        return {
            'BASE_PATH':             self.prop_base_path  if self.prop_base_path else None,
            'FILE_NAME':             self.prop_file_name  if self.prop_file_name else None,
            'NEW_SCENE_OBJECT_NAME': self.prop_new_scene_object_name if self.prop_new_scene_object_name else None,
            'MODEL_NAME':            self.prop_model_name if not self.prop_model_name == 'None'  else None,
            'MPART_NAME':            self.prop_mpart_name if not self.prop_mpart_name == '*'     else None,
            'NEW_MODEL_NAME':        self.prop_new_model_name if self.prop_new_model_name else None,
            'NEW_MPART_NAME':        self.prop_new_mpart_name if self.prop_new_mpart_name else None,
            'NEW_MPART_PRIMITIVE':   self.prop_new_mpart_mesh if self.prop_new_mpart_mesh else None,
            'ACTION_TYPE':           self.prop_action_type, 
            'ACTION_KIND':           self.prop_action_kind,
            'ACTION_NAME':           self.prop_action_name if not self.prop_action_name == 'None' else None,
            'NEW_ACTION_NAME':       self.prop_new_action_name if self.prop_new_action_name   else None, 
            'IMAGE_TYPE':            self.prop_image_type, 
            'NEW_IMAGE_NAME':        self.prop_new_image_name if self.prop_new_image_name else None,
            'TEX_NODE_IMAGE_NAME':   self.prop_tex_node_image_name if not self.prop_tex_node_image_name == 'None' else None, 
        }


    def get_prop_model_name_enum_items(self, context):
        out = []
        for MODEL_NAME in get_scene_model_names():
            item = (MODEL_NAME, MODEL_NAME, 'model')
            out.append(item)
        if not out:
            out.append(('None', '-', 'empty'))
        return out

    def get_prop_mpart_name_enum_items(self, context):
        out = [('*', '*', 'all selected')]
        if not self.prop_mpart_selection:
            return out 

        MODEL_NAME = self.prop_model_name
        for MPART_NAME in get_model_mpart_names(MODEL_NAME):
            item = (MPART_NAME, MPART_NAME, 'mpart')
            out.append(item)
        return out

    def get_prop_action_type_enum_items(self, context):
        out = [
            (ACTION_TYPE_MPART, ACTION_TYPE_MPART, 'mpart action type'),
            (ACTION_TYPE_MODEL, ACTION_TYPE_MODEL, 'model action type'),
            (ACTION_TYPE_SCENE, ACTION_TYPE_SCENE, 'scene action type') 
        ]
        return out

    def get_prop_action_kind_enum_items(self, context):
        out = [
            (ACTION_KIND_LM, ACTION_KIND_LM, 'lm action kind'), 
            (ACTION_KIND_LRSM, ACTION_KIND_LRSM, 'lrsm action kind'), 
            (ACTION_KIND_ANIMATION, ACTION_KIND_ANIMATION, 'animation action kind')
        ]
        return out

    def get_prop_action_name_enum_items(self, context):

        actions = []
        if self.prop_action_type == ACTION_TYPE_SCENE:
            actions = find_action(ACTION_TYPE_SCENE, self.prop_action_kind, None)

        if self.prop_action_type == ACTION_TYPE_MODEL:
            actions = find_action(ACTION_TYPE_MODEL, self.prop_action_kind, None, self.prop_model_name)

        if self.prop_action_type == ACTION_TYPE_MPART:
            actions = find_action(
                                ACTION_TYPE_MPART, 
                                self.prop_action_kind, 
                                None, 
                                self.prop_model_name, 
                                self.prop_mpart_name if not self.prop_mpart_name == '*' else None)

        out = []
        seen = set()

        for action in actions:
            name = action.name.split('-')[2]
            if name in seen:
                continue 
            seen.add(name)
            out.append((name, name, 'action name'))

        if not out:
            out.append(('None', '-', 'empty'))
        return out 

    def get_prop_image_type_enum_items(self, context):
        out = [
            (IMAGE_TYPE_MPART_DIFFUSE, IMAGE_TYPE_MPART_DIFFUSE, 'mpart diffuse'),
            (IMAGE_TYPE_MODEL_DIFFUSE, IMAGE_TYPE_MODEL_DIFFUSE, 'model diffuse'), 
            (IMAGE_TYPE_SCENE_DIFFUSE, IMAGE_TYPE_SCENE_DIFFUSE, 'scene diffuse'), 
        ]
        return out 

    def get_prop_tex_node_image_name_enum_items(self, context):

        MODEL_NAME = self.prop_model_name
        MPART_NAME = self.prop_mpart_name if not self.prop_mpart_name == '*' else None
        images = find_images(IMAGE_TYPE_MPART_DIFFUSE, None, MODEL_NAME, MPART_NAME)

        out = []
        seen = set()
        for img in images:
            name = img.name.split('-')[1]
            if name in seen:
                continue 
            seen.add(name)
            out.append((name, name, 'image name'))

        if not out:
            out.append(('None', '-', 'empty'))
        return out 

    def get_prop_new_mpart_mesh_enum_items(self, context):
        out = [
                ('CUBE', 'CUBE', 'cube')
                ]
        return out 

    def clean_up_property_values(self):
        self.prop_export = False
        self.prop_clean_up = False
        self.prop_create_scene_object = False
        self.prop_create_new_model = False
        self.prop_create_new_mpart = False
        self.prop_mpart_selection = False
        self.prop_mpart_name = '*'
        self.prop_smart_uv_project = False
        self.prop_set_action = False
        self.prop_create_action = False
        self.prop_create_image = False
        self.prop_create_material = False

    prop_export              = bpy.props.BoolProperty(name='export', default=False)
    prop_clean_up            = bpy.props.BoolProperty(
                                            name='clean up', 
                                            default=False,
                                            description='remove actions and images with zero users')

    prop_base_path           = bpy.props.StringProperty(name='path')
    prop_file_name           = bpy.props.StringProperty(name='filename')

    prop_create_scene_object = bpy.props.BoolProperty(
                                            name='create scene object', 
                                            default=False,
                                            description='create a scene SM object')

    prop_new_scene_object_name   = bpy.props.StringProperty(name='scene object name')

    prop_model_name          = bpy.props.EnumProperty(
                                            name='current model',
                                            items=get_prop_model_name_enum_items)
    
    prop_mpart_selection     = bpy.props.BoolProperty(
                                            name='per mpart selection', 
                                            default=False,
                                            description='enable per mpart selection else select all')

    prop_mpart_name          = bpy.props.EnumProperty(
                                            name='current mpart',
                                            items=get_prop_mpart_name_enum_items)


    prop_create_new_model   = bpy.props.BoolProperty(
                                            name='create model',
                                            default=False,
                                            description='create new model')
    
    prop_new_model_name      = bpy.props.StringProperty(name='new model name')


    prop_create_new_mpart    = bpy.props.BoolProperty(
                                            name='create mpart',
                                            default=False,
                                            description='create new mpart')

    prop_new_mpart_name      = bpy.props.StringProperty(name='new mpart name')
    prop_new_mpart_mesh      = bpy.props.EnumProperty(
                                            name='new mpart mesh',
                                            items=get_prop_new_mpart_mesh_enum_items)

    prop_smart_uv_project    = bpy.props.BoolProperty(
                                            name='smart uv project', 
                                            default=False,
                                            description='smart uv project selected model')

    prop_action_type         = bpy.props.EnumProperty(
                                            name='action type',
                                            items=get_prop_action_type_enum_items,
                                            description='action type')

    prop_action_kind         = bpy.props.EnumProperty(
                                            name='action kind',
                                            items=get_prop_action_kind_enum_items,
                                            description='action kind')
    
    prop_set_action          = bpy.props.BoolProperty(
                                            name='mark action current', 
                                            default=False,
                                            description='make action curent for model/mpart')

    prop_action_name         = bpy.props.EnumProperty(
                                            name='available actions', 
                                            items=get_prop_action_name_enum_items,
                                            description='available actions'
                                            )

    prop_create_action       = bpy.props.BoolProperty(
                                        name='create action', 
                                        default=False,
                                        description='create new action')

    prop_new_action_name     = bpy.props.StringProperty(name='new action name')

    prop_create_image        = bpy.props.BoolProperty(name='create image', default=False)
    prop_image_type          = bpy.props.EnumProperty(name='image type', items=get_prop_image_type_enum_items)
    prop_new_image_name      = bpy.props.StringProperty(name='new image name')

    prop_create_material     = bpy.props.BoolProperty(name='create materials', default=False)
    prop_tex_node_image_name = bpy.props.EnumProperty(
                                            name='tex node image', 
                                            items=get_prop_tex_node_image_name_enum_items,
                                            description='image for image texture node while creating materials'
                                            )

    def raise_error(self, msg):
        self.report({'ERROR'}, msg)
        return True

    def raise_warning(self, msg):
        self.report({'WARNING'}, msg)
        return True

    def validate_data(self, data, required):
        return all(data[key] != None for key in required)

    def operation_smart_uv_project(self, data):
        required = {'MODEL_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("model name empty") and False

        objects = find_meshed_objects(EXPORT_OBJECT, data['MODEL_NAME'], data['MPART_NAME'])

        for obj in objects:
            obj.select = True
            bpy.ops.uv.smart_project()
            obj.select = False

        return True

    def operation_set_action_scene(self, data):
        required = {'ACTION_TYPE', 'ACTION_KIND', 'ACTION_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        objs = find_meshed_objects(SCENE_OBJECT)
        if not objs:
            return self.raise_error('scene object not found') and False

        obj = objs[0]
        if not obj.animation_data:
            obj.animation_data_create()

        actions = find_action(data['ACTION_TYPE'], data['ACTION_KIND'], data['ACTION_NAME'])
        if not actions:
            return self.raise_error('%s action not found' % (data['ACTION_NAME'])) and False

        obj.animation_data.action = actions[0]
        return True

    def operation_set_action_model(self, data):
        required = {'ACTION_TYPE', 'ACTION_KIND', 'ACTION_NAME', 'MODEL_NAME'}
        
        if not self.validate_data(data, required):
            return self.raise_error('invalid form') and False

        objs = find_meshed_objects(MODEL_OBJECT, data['MODEL_NAME'])
        if not objs:
            return self.raise_error('%s model not found' % (data['MODEL_NAME'])) and False

        obj = objs[0]
        if not obj.animation_data:
            obj.animation_data_create()

        actions = find_action(data['ACTION_TYPE'], data['ACTION_KIND'], data['ACTION_NAME'], data['MODEL_NAME'])
        if not actions:
            return self.raise_error('%s action not found for %s' % (data['ACTION_NAME'], data['MODEL_NAME'])) and False

        obj.animation_data.action = actions[0]
        return True

    def operation_set_action_mpart(self, data):
        required = {'ACTION_TYPE', 'ACTION_KIND', 'ACTION_NAME', 'MODEL_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        objects = find_meshed_objects(EXPORT_OBJECT, data['MODEL_NAME'], data['MPART_NAME'])

        if not objects:
            return raise_error('no objects found') and False

        for obj in objects:
            if not obj.animation_data:
                obj.animation_data_create()

            MPART_NAME = obj.name.split('-')[-1]
            actions = find_action(data['ACTION_TYPE'], data['ACTION_KIND'], data['ACTION_NAME'], data['MODEL_NAME'], MPART_NAME)
            if not actions:
                return self.raise_error('action not found for %s' % (MPART_NAME)) and False
            obj.animation_data.action = actions[0]

        return True 

    def operation_set_action(self, data):
        if data['ACTION_TYPE'] == ACTION_TYPE_SCENE:
            return self.operation_set_action_scene(data)
        if data['ACTION_TYPE'] == ACTION_TYPE_MODEL:
            return self.operation_set_action_model(data)
        if data['ACTION_TYPE'] == ACTION_TYPE_MPART:
            return self.operation_set_action_mpart(data)

        return self.raise_error('invalid action type') and False

    def operation_create_action_scene(self, data):
        required = {'ACTION_TYPE', 'ACTION_KIND', 'NEW_ACTION_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        action_name = '-'.join([data['ACTION_TYPE'], data['ACTION_KIND'], data['NEW_ACTION_NAME']])

        if not is_action_present(action_name):
            bpy.data.actions.new(name=action_name)
        else:
            self.raise_warning('action already exists')
        return True

    def operation_create_action_model(self, data):
        required = {'ACTION_TYPE', 'ACTION_KIND', 'NEW_ACTION_NAME', 'MODEL_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        action_name = '-'.join([data['ACTION_TYPE'], data['ACTION_KIND'], data['NEW_ACTION_NAME'], data['MODEL_NAME']])

        if not is_action_present(action_name):
            bpy.data.actions.new(name=action_name)
        else:
            self.raise_warning('action already exists')
        return True

    def operation_create_action_mpart(self, data):
        required = {'ACTION_TYPE', 'ACTION_KIND', 'NEW_ACTION_NAME', 'MODEL_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        objects = find_meshed_objects(EXPORT_OBJECT, data['MODEL_NAME'], data['MPART_NAME'])

        if not objects:
            return raise_error('no objects found') and False

        for obj in objects:
            MPART_NAME = obj.name.split('-')[-1]
            action_name = [data['ACTION_TYPE'], data['ACTION_KIND'], data['NEW_ACTION_NAME'], data['MODEL_NAME'], MPART_NAME]
            action_name = '-'.join(action_name)
            if not is_action_present(action_name):
                bpy.data.actions.new(name=action_name)
            else:
                self.raise_warning('action already exists')

        return True 

    def operation_create_action(self, data):
        if data['ACTION_TYPE'] == ACTION_TYPE_SCENE:
            return self.operation_create_action_scene(data)
        if data['ACTION_TYPE'] == ACTION_TYPE_MODEL:
            return self.operation_create_action_model(data)
        if data['ACTION_TYPE'] == ACTION_TYPE_MPART:
            return self.operation_create_action_mpart(data)

        return self.raise_error('invalid action type') and False


    def operation_prop_create_image_scene(self, data):
        required = {'IMAGE_TYPE', 'NEW_IMAGE_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        image_name = '-'.join([data['IMAGE_TYPE'], data['NEW_IMAGE_NAME']])
        if not is_image_present(image_name):
            bpy.data.images.new(name=image_name, width=1024, height=1024)
        else:
            self.raise_warning('image already exists')
        return True

    def operation_prop_create_image_model(self, data):
        required = {'IMAGE_TYPE', 'NEW_IMAGE_NAME', 'MODEL_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        image_name = '-'.join([data['IMAGE_TYPE'], data['NEW_IMAGE_NAME'], data['MODEL_NAME']])
        if not is_image_present(image_name):
            bpy.data.images.new(name=image_name, width=1024, height=1024)
        else:
            self.raise_warning('image already exists')
        return True

    def operation_prop_create_image_mpart(self, data):
        required = {'IMAGE_TYPE', 'NEW_IMAGE_NAME', 'MODEL_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        objects = find_meshed_objects(EXPORT_OBJECT, data['MODEL_NAME'], data['MPART_NAME'])

        if not objects:
            return self.raise_error('mparts not found for %s', data['MODEL_NAME']) and False

        for obj in objects:
            MPART_NAME = obj.name.split('-')[2]
            image_name = [data['IMAGE_TYPE'], data['NEW_IMAGE_NAME'], data['MODEL_NAME'], MPART_NAME]
            image_name = '-'.join(image_name)
            if not is_image_present(image_name):
                bpy.data.images.new(name=image_name, width=1024, height=1024)
            else:
                self.raise_warning('image already exists')
        return True

    def operation_prop_create_image(self, data):
        if data['IMAGE_TYPE'] == IMAGE_TYPE_SCENE_DIFFUSE:
            return self.operation_prop_create_image_scene(data)

        if data['IMAGE_TYPE'] == IMAGE_TYPE_MODEL_DIFFUSE:
            return self.operation_prop_create_image_model(data)

        if data['IMAGE_TYPE'] == IMAGE_TYPE_MPART_DIFFUSE:
            return self.operation_prop_create_image_mpart(data)

        return self.raise_error('invalid image type') and False

    def operation_prop_create_material(self, data):
        required = {'MODEL_NAME', 'TEX_NODE_IMAGE_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        objects = find_meshed_objects(EXPORT_OBJECT, data['MODEL_NAME'], data['MPART_NAME'])

        if not objects:
            return self.raise_error('objects not found') and False

        for obj in objects:
            _, MODEL_NAME, MPART_NAME = obj.name.split('-')
            material_name = '%s %s-MATERIAL' % (MODEL_NAME, MPART_NAME)

            if not obj.data.materials:
                mat = bpy.data.materials.new(name=material_name)
                mat.use_nodes = True
                obj.data.materials.append(mat)
                nodes = mat.node_tree.nodes
                image_tex_node = nodes.new('ShaderNodeTexImage')
            else:
                mat = obj.data.materials[0]
                nodes = mat.node_tree.nodes
                image_tex_node = nodes['Image Texture']

            images = find_images(IMAGE_TYPE_MPART_DIFFUSE, data['TEX_NODE_IMAGE_NAME'], MODEL_NAME, MPART_NAME)
            if not images:
                return self.raise_error(
                            'could not find image %s for %s' % (
                                        data['TEX_NODE_IMAGE_NAME'], 
                                        MPART_NAME)
                            ) and False

            image_tex_node.image = images[0]

        return True 

    def operation_prop_export(self, data):
        required = {'BASE_PATH', 'FILE_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error("invalid form") and False

        return export_scene(data['BASE_PATH'], data['FILE_NAME'])

    def operation_prop_clean_up(self, data):
        for img in bpy.data.images:
            if img.users == 0 or not img.has_data:
                bpy.data.images.remove(img)

        for action in bpy.data.actions:
            if action.users == 0:
                bpy.data.actions.remove(action)
        return True

    def operation_prop_create_new_model(self, data, context):
        required = {'NEW_MODEL_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error('invalid model name') and False

        if find_meshed_objects(MODEL_OBJECT, data['NEW_MODEL_NAME']):
            return self.raise_error('model %s already present' % (data['NEW_MODEL_NAME']))

        object_name = '%s-%s' % (MODEL_OBJECT, data['NEW_MODEL_NAME'])

        if '__BEGL__EMPTY_MESH__' not in bpy.data.meshes:
            bpy.data.meshes.new('__BEGL__EMPTY_MESH__')

        new_mesh = bpy.data.meshes['__BEGL__EMPTY_MESH__']
        new_obj  = bpy.data.objects.new(object_name, new_mesh)
        context.scene.objects.link(new_obj)
        return True 

    def operation_prop_create_new_mpart(self, data, context):
        required = {'MODEL_NAME', 'NEW_MPART_NAME', 'NEW_MPART_PRIMITIVE'}
        
        if not self.validate_data(data, required):
            return self.raise_error('invalid form') and False

        if not find_meshed_objects(MODEL_OBJECT, data['MODEL_NAME']):
            return self.raise_error('invalid model name %s', data['MODEL_NAME'])

        if find_meshed_objects(EXPORT_OBJECT, data['MODEL_NAME'], data['NEW_MPART_NAME']):
            return self.raise_error('mpart already exists')

        MPART_NAME = '%s-%s-%s' % (EXPORT_OBJECT, data['MODEL_NAME'], data['NEW_MPART_NAME'])

        if data['NEW_MPART_PRIMITIVE'] == 'CUBE':
            bpy.ops.mesh.primitive_cube_add()
            context.scene.objects.active.name = MPART_NAME
            context.scene.objects.active.data.name = '%s-MESH' % (MPART_NAME)

        return True

    def operation_prop_create_scene_object(self, data, context):
        required = {'NEW_SCENE_OBJECT_NAME'}

        if not self.validate_data(data, required):
            return self.raise_error('invalid form') and False

        if find_meshed_objects(SCENE_OBJECT):
            return self.raise_error('scene object already present')

        object_name = '%s-%s' % (SCENE_OBJECT, data['NEW_SCENE_OBJECT_NAME'])

        if '__BEGL__EMPTY_MESH__' not in bpy.data.meshes:
            bpy.data.meshes.new('__BEGL__EMPTY_MESH__')

        new_mesh = bpy.data.meshes['__BEGL__EMPTY_MESH__']
        new_obj  = bpy.data.objects.new(object_name, new_mesh)
        context.scene.objects.link(new_obj)
        return True 

    def begl_execute(self, context):
        data = self.get_property_values()

        if self.prop_clean_up and not self.operation_prop_clean_up(data):
            return 

        if self.prop_create_scene_object and not self.operation_prop_create_scene_object(data, context):
            return

        if self.prop_create_new_model and not self.operation_prop_create_new_model(data, context):
            return

        if self.prop_create_new_mpart and not self.operation_prop_create_new_mpart(data, context):
            return 

        if self.prop_smart_uv_project and not self.operation_smart_uv_project(data):
            return

        if self.prop_create_action and not self.operation_create_action(data):
            return

        if self.prop_create_image and not self.operation_prop_create_image(data):
            return

        if self.prop_set_action and not self.operation_set_action(data):
            return

        if self.prop_create_material and not self.operation_prop_create_material(data):
            return

        if self.prop_export and not self.operation_prop_export(data):
            return

    def execute(self, context):
        self.begl_execute(context)
        return {'FINISHED'}
 
    def bgl_invoke(self):
        self.clean_up_property_values()

    def invoke(self, context, event): # Used for user interaction
        self.bgl_invoke()
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

for action in bpy.data.actions:
    print (action.name)

bpy.utils.register_class(BEGLToolbar)
