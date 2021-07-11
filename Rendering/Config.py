import pyrender
from typing import Union,List,Dict,Tuple
import numpy as np

class Config:
    camera_folder : str = "."
    bg_color : List[float] = [0.25,0.25,0.25,0]
    ambient_light : List[float] = [0.3,0.3,0.3]
    use_custom_shader : bool = False
    
    @staticmethod
    def Viewer(scene:pyrender.Scene) -> pyrender.Viewer:
        return pyrender.Viewer(scene,render_flags={"all_solid":True},viewer_flags={"use_direct_lighting":True})
    
    @staticmethod
    def Scene() -> pyrender.Scene:
        return pyrender.Scene(bg_color = Config.bg_color, ambient_light = Config.ambient_light)
        
