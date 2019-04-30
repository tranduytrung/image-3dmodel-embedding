import re
from .material import Material

class MTLParser:
    re_material_name = re.compile(r'newmtl\s+(?P<name>\w+)')
    re_ambien = re.compile(r'Ka\s+(?P<r>\d+(\.\d*)?)\s+(?P<g>\d+(\.\d*)?)\s+(?P<b>\d+(\.\d*)?)')
    re_diffuse = re.compile(r'Kd\s+(?P<r>\d+(\.\d*)?)\s+(?P<g>\d+(\.\d*)?)\s+(?P<b>\d+(\.\d*)?)')
    re_specular = re.compile(r'Ks\s+(?P<r>\d+(\.\d*)?)\s+(?P<g>\d+(\.\d*)?)\s+(?P<b>\d+(\.\d*)?)')
    re_shininess = re.compile(r'Ns\s+(?P<value>\d+(\.\d*)?)')
    re_illumination_mode = re.compile(r'illum\s+(?P<value>\d+)')
    
    @staticmethod
    def from_string(data):
        start = 0
        match = MTLParser.re_material_name.search(data, start)
        materials = []
        while match is not None:
            start = match.end()
            name = match.groupdict()['name']
            match = MTLParser.re_ambien.search(data, start)
            m_dict = match.groupdict()
            ka = (float(m_dict['r']), float(m_dict['g']), float(m_dict['b']))
            match = MTLParser.re_diffuse.search(data, start)
            m_dict = match.groupdict()
            kd = (float(m_dict['r']), float(m_dict['g']), float(m_dict['b']))
            match = MTLParser.re_illumination_mode.search(data, start)
            m_dict = match.groupdict()
            illum = int(m_dict['value'])
            if illum == 2:
                match = MTLParser.re_specular.search(data, start)
                m_dict = match.groupdict()
                ks = (float(m_dict['r']), float(m_dict['g']), float(m_dict['b']))
                match = MTLParser.re_shininess.search(data, start)
                m_dict = match.groupdict()
                ns = float(m_dict['value'])
                material = Material(name, ka, kd, ks, ns, illum=2)
            else:
                material = Material(name, ka, kd, illum=1)
            materials.append(material)

            match = MTLParser.re_material_name.search(data, start)
        return materials
            
    @staticmethod
    def from_file(filename):
        with open(filename, 'rt') as reader:
            data = reader.read()
        
        return MTLParser.from_string(data)    