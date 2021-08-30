from typing import Dict
from helper import AllOf
import json
import sys
import os
from logger.utils import get_logger


def build_config_file(path: str, folder:str, replace: dict = {}):
    try:
        fpath = folder + "/" + path
        with open(fpath, 'r') as file:
            json_text = file.read()
        obj = json.loads(json_text)
        #allows us to load files from other directories
        folder = os.path.dirname(fpath)

        return build_config(obj,folder,replace)
    except Exception  as e:
        print("Error parsing config file : " + folder + "/" + path)
        print(type(e).__name__)
        print(e)
        import traceback
        print(traceback.print_tb(e.__traceback__))
        sys.exit(0)

def concat_json(elements,folder,replace):
    if(len(elements) == 0 or not isinstance(elements,list)):
        raise ValueError("Cant concat empty or non array")
    elements = list(map(lambda item:build_config(item,folder,replace),elements))

    first = elements[0]
    if(isinstance(first,str)):
        return ''.join(map(str,elements))
    if(isinstance(first,list)):
        return sum(elements,[])

    raise ValueError(f"Cant concat elements of type {type(first).__name__}")

def build_config(json : object, folder : str, replace: dict = {}):
    """
    build_config method to put together single json object for full run (which then will be saved)

    """
    if( isinstance(json, dict) and "comment" in json):
        del json["comment"]
    if(isinstance(json,dict)):
        #special case if we are due to insert another json file here
        file = json.pop("insert", None)
        if(isinstance(file,str) and file.startswith("replace:")):
            file = replace[file[8:]]
        if(file is not None):
            replaceNext = replace.copy()
            for k,v in json.items():
                replaceNext[k] = build_config(v,folder,replace)

            result = build_config_file(file,folder,replaceNext)
            return result
        concat = json.pop("concat",None)
        if(concat is not None):
            return concat_json(concat,folder,replace)

        return dict(map(lambda item: (item[0], build_config(item[1],folder,replace)), json.items()))
    if(isinstance(json,list)):
        return list(map(lambda item:build_config(item,folder,replace),json))
    #an inserted json file may take arguments from the parent one
    if(isinstance(json,str) and json.startswith("replace:")):
        return replace[json[8:]]

    return json




class ConfigObject(object):


    def __init__(self, key_value):
        self.key_value : Dict[str,object]
        self.runner = None
        self.runner = key_value.get("runner",self)
        self.key_value = key_value
        if self.runner is not self:
            self.py_logger = self.runner.py_logger
        else:
            self.py_logger = get_logger(self.runner.name)
        for key,value in key_value.items():
            self.process(value,key)


    def get_class_for(self, name:str, classname:str):
        """
        get_class_for returns the class using the name for classname
        :param name: name of the field instance is to be assigned to
        :param classname: classname of the class
        """
        raise ValueError("get_class_for is not overwritten by class {} for field {} with type {} ".format(type(self),name,classname))

    def postprocess(self, obj:object, key:str):
        if(isinstance(obj,list)):
            return AllOf(obj)
        return obj

    def process(self, obj:object, key:str):

        if(isinstance(obj, dict)) and 'type' in obj:
            obj["runner"] = self.runner
            classtype = self.get_class_for(key, obj["type"])
            obj = classtype(obj)
            obj.runner = self.runner

        if(isinstance(obj,list)):
            obj = list(map(lambda item:self.process(item,key),obj))


        obj = self.postprocess(obj, key)

        setattr(self,key,obj)

        return obj

    def c_get(self, name:str, default:object = None) -> str:
        return self.key_value.get(name,default)

    def c_get_s(self, name: str, default: str = None) -> str:
        value = self.c_get(name,None)
        if( value is None):
            return default
        return str(value)

    def c_get_i(self, name: str, default: int = None) -> int:
        value = self.c_get(name,None)
        if(value is None):
            return default
        return int(value)

