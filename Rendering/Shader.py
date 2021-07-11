import pyrender
import os
folder = os.path.dirname(os.path.abspath(__file__))

class ShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(folder+"/shaders/mesh.vert", folder+"/shaders/mesh.frag", defines=defines)
        return self.program
    def clear(self):
        del self.program
        self.program = None