import os

os.environ["PANDA_PRC_DIR"] = os.path.dirname(os.path.realpath(__file__))
os.environ["PANDA_PRC_PATH"] = os.path.dirname(os.path.realpath(__file__))

from threading import Lock
from direct.showbase.ShowBase import ShowBase
from panda3d.core import VBase4, VBase3, VBase2
from panda3d.core import Filename, loadPrcFileData
from panda3d.core import DirectionalLight, AmbientLight, PointLight, PerspectiveLens, AntialiasAttrib
import numpy as np
import PIL.Image

class Panda3DRenderer:
    __lock = Lock()

    def __init__(self, cad_file=None, output_size=(512, 512), light_on=True, cast_shadow=True):
        # acquire lock since showbase cannot be created twice
        Panda3DRenderer.__lock.acquire()

        # set output size and init the base
        loadPrcFileData('', f'win-size {output_size[0]} {output_size[1]}') 
        base = ShowBase(windowType='offscreen')

        # coordinate for normalized and centered object
        obj_node = base.render.attach_new_node('normalized_obj')

        # ambient
        alight = AmbientLight('alight')
        alight.set_color(VBase4(0.5, 0.5, 0.5, 1.0))
        alnp = base.render.attachNewNode(alight)
        base.render.setLight(alnp)

        # directional light for ambient
        dlight1 = DirectionalLight('dlight1')
        dlight1.set_color(VBase4(0.235, 0.235, 0.235, 1.0))
        dlnp1 = base.render.attach_new_node(dlight1)
        dlnp1.set_pos(-2, 3, 1)
        dlnp1.look_at(obj_node)
        base.render.set_light(dlnp1)

        # point light for ambient
        plight1 = PointLight('plight1')
        plight1.set_color(VBase4(1.75, 1.75, 1.75, 1.0))
        plight1.setAttenuation((1,1,1))
        plnp1 = base.render.attach_new_node(plight1)
        plnp1.set_pos(0, 0, 3)
        plnp1.look_at(obj_node)
        base.render.set_light(plnp1)

        plight2 = PointLight('plight2')
        plight2.set_color(VBase4(1.5, 1.5, 1.5, 1.0))
        plight2.setAttenuation((1,0,1))
        plnp2 = base.render.attach_new_node(plight2)
        plnp2.set_pos(0, -3, 0)
        plnp2.look_at(obj_node)
        base.render.set_light(plnp2)

        dlight2 = DirectionalLight('dlight2')
        dlight2.set_color(VBase4(0.325, 0.325, 0.325, 1.0))
        dlnp2 = base.render.attach_new_node(dlight2)
        dlnp2.set_pos(-1, 1, -1.65)
        dlnp2.look_at(obj_node)
        base.render.set_light(dlnp2)

        dlight3 = DirectionalLight('dlight3')
        dlight3.set_color(VBase4(0.15, 0.15, 0.15, 1.0))
        dlnp3 = base.render.attach_new_node(dlight3)
        dlnp3.set_pos(-2.5, 2.5, 2.0)
        dlnp3.look_at(obj_node)
        base.render.set_light(dlnp3)
        if cast_shadow:
            lens = PerspectiveLens()
            dlight3.set_lens(lens)
            dlight3.set_shadow_caster(True, 1024, 1024)

        dlight4 = DirectionalLight('dlight4')
        dlight4.set_color(VBase4(0.17, 0.17, 0.17, 1.0))
        dlnp4 = base.render.attach_new_node(dlight4)
        dlnp4.set_pos(1.2, -2.0, 2.5)
        dlnp4.look_at(obj_node)
        base.render.set_light(dlnp4)
        if cast_shadow:
            lens = PerspectiveLens()
            dlight4.set_lens(lens)
            dlight4.set_shadow_caster(True, 1024, 1024)

        self.direct_node = direct_node = base.render.attach_new_node('direct_light')
        dlnp2.reparent_to(direct_node)
        dlnp3.reparent_to(direct_node)
        dlnp4.reparent_to(direct_node)
        
        # auto shader for shadow
        if cast_shadow:
            base.render.setShaderAuto()
        # no culling
        base.render.set_two_sided(True)
        # anti-alias
        base.render.setAntialias(AntialiasAttrib.MMultisample, 8)
        # init camera position
        self.coverage = 0.5
        # the default clear color
        self.clear_color = (0.0 ,0.0 , 0.0, 0.0)
        # translate in rendered image
        self.obj_translate = (0, 0)
        # light rotation
        self.light_hpr = (0, 0, 0)
        # object rotation
        self.obj_hpr = (0, 0, 0)

        self.base = base
        self.obj = None
        self.obj_node = obj_node
        self.cast_shadow = cast_shadow
        self.camera = base.camera
        if cad_file is not None:
            self.set_obj(cad_file)            

        if not light_on:
            base.render.set_light_off()

    @property
    def clear_color(self):
        b4 = self._clear_color
        return b4.get_x(), b4.get_y(), b4.get_z(), b4.get_w()

    @clear_color.setter
    def clear_color(self, value):
        self._clear_color = VBase4(*value)

    @property
    def light_hpr(self):
        hpr = self._light_hpr
        return hpr.get_x(), hpr.get_y(), hpr.get_z()

    @light_hpr.setter
    def light_hpr(self, value):
        self._light_hpr = VBase3(*value)

    @property
    def obj_hpr(self):
        hpr = self._obj_hpr
        return hpr.get_x(), hpr.get_y(), hpr.get_z()

    @obj_hpr.setter
    def obj_hpr(self, value):
        self._obj_hpr = VBase3(*value)
        
    @property
    def obj_translate(self):
        xy = self._obj_translate
        return xy.get_x(), xy.get_y()

    @obj_translate.setter
    def obj_translate(self, value):
        self._obj_translate = VBase2(*value)
        
    @property
    def obj_color(self):
        b4 = self.obj.get_color()
        return b4.get_x(), b4.get_y(), b4.get_z(), b4.get_w()

    @obj_color.setter
    def obj_color(self, value):
        self.obj.set_color(*value)

    def get_camera_radius(self):
        r = 1
        c = self.coverage
        lens = self.base.camLens
        tan_cam = lens.get_film_size()[1] / 2 / lens.get_focal_length()
        # tan_cam = lens.get_vfov() / 2 / lens.get_near()
        return r/(c*tan_cam)

    def get_dlight_pos(self):
        theta = self.dlight_params['theta']
        phi = self.dlight_params['phi']
        radius = self.get_camera_radius()
        x = radius*np.sin(theta)*np.cos(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = np.absolute(radius*np.cos(theta))
    
        return VBase3(x, y, z)

    def set_obj(self, obj_path):
        obj_node = self.obj_node
        cast_shadow = self.cast_shadow

        if self.obj is not None:
            self.obj.remove_node()

        # Convert that to panda's unix-style notation.
        filepath = Filename.fromOsSpecific(obj_path).getFullpath()
        # load model and add to render root
        obj = base.loader.load_model(filepath, noCache=True)
        obj.reparent_to(obj_node)
        # scale to uinit sphere
        obj_bounds = obj_node.getBounds()
        obj.set_scale(1/obj_bounds.radius)
        obj_bounds = obj_node.getBounds()
        obj.set_pos(-obj_bounds.center)
        if cast_shadow:
            obj.set_depth_offset(-1)

        self.obj = obj

    def render(self, binary=True):
        base = self.base
        # context
        base.win.setClearColor(self._clear_color)
        base.camera.set_pos(VBase3(0, -self.get_camera_radius(), 0))
        self.obj_node.set_hpr(self._obj_hpr)
        self.direct_node.set_hpr(self._light_hpr)

        # redner
        base.graphics_engine.render_frame()
        tex = base.win.get_screenshot()

        # export
        bytes_image = bytes(tex.get_ram_image_as('RGBA'))
        pil_image = PIL.Image.frombytes('RGBA', base.get_size(), bytes_image)
        pil_image = pil_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        if tuple(self.obj_translate) != (0, 0):
            np_clear_color = np.array(self._clear_color*255, dtype=int)
            translated_image = PIL.Image.new('RGBA', pil_image.size, tuple(np_clear_color))
            np_obj_translate = np.array(self.obj_translate, dtype=int)
            translated_image.paste(pil_image, tuple(np_obj_translate))
            pil_image = translated_image

        if not binary:
            return pil_image

        bin_image = PIL.Image.new('1', pil_image.size)
        bin_image.paste(1, None, pil_image)

        return pil_image, bin_image

    def random_context(self, light=True, obj_rotation=True, obj_color=True,
                       obj_translate=True, coverage=[0.1, 0.5]):
        r = 1  # unit sphere

        if coverage is not None:
            if hasattr(coverage, '__len__'):
                if len(coverage) == 2:
                    self.coverage = np.random.uniform(coverage[0], coverage[1])
                else:
                    self.coverage = coverage[0]
            elif isinstance(coverage, (int, float)):
                self.coverage = coverage
            else:
                raise ValueError()

        if obj_translate:
            self.obj_translate = np.random.uniform(-0.5 + self.coverage/2, 0.5 - self.coverage/2, size=2) \
                                     * self.base.get_size()
            
        # light
        if light:
            self.light_hpr = np.random.uniform(
                -180, 180, size=3)

        # obj rotate
        if obj_rotation:
            self.obj_hpr = np.random.uniform(
                -180, 180, size=3)

        if obj_color:
            self.obj_color = (*np.random.uniform(size=3), 1)
            
    def close(self):
        self.base.destroy()
        Panda3DRenderer.__lock.release()
        
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        self.close()