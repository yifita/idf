from task.task import Task
from typing import Tuple

class SetNetwork(Task):
    usable_keys : Tuple[str] = ("network", "data", "loss")

    def __init__(self, config):
        super().__init__(config)

    def filter(self, name:str):
        return name.startswith(self.usable_keys)

    def get_class_for(self, name:str, classname:str):
        #all subclasses are of type network
        return self.runner.get_class_for("network", classname)

    def process(self, obj:object, key:str):
        # override default config, lazy loading
        if(not self.filter(key)):
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

    def __call__(self):
        for key in self.key_value.keys():
            if(not self.filter(key)):
                continue
            value = self.key_value[key]
            #in case we are recreating a network here
            if(isinstance(value, dict) and 'type' in value):
                value["runner"] = self.runner
                value = self.get_class_for("", value['type'])(value)

            self.runner.py_logger.info(f"Setting Network property {key} to value {value}")
            keys =  key.split(".")

            obj = self.runner
            for subkey in keys[0:-1]:
                obj = getattr(obj, subkey, None)
                if(obj is None):
                    break

            if obj is None:
                self.runner.py_logger.info(f"Key not found {subkey} from {key}")

            setattr(obj, keys[-1], value)