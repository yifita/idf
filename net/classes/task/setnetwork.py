from task.task import Task

class SetNetwork(Task):

    def __init__(self, config):
        super().__init__(config)

    def get_class_for(self, name:str, classname:str):
        #all subclasses are of type network
        return self.runner.get_class_for("network", classname)

    def process(self, obj:object, key:str):
        # override default config, lazy loading
        if(key[:len("network")] != "network"):
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
            if(key[:len("network")] != "network"):
                continue
            value = self.key_value[key]
            assert(isinstance(value, dict) and 'type' in value)
            value["runner"] = self.runner
            value = self.get_class_for("", value['type'])(value)

            self.runner.py_logger.info(f"Setting Network property {key} to value {value}")
            keys =  key.split(".")
            assert(keys[0] == "network")
            obj = self.runner.network

            for subkey in keys[1:-1]:
                obj = getattr(obj, subkey, None)
                if(obj is None):
                    break

            if obj is None:
                self.runner.py_logger.info(f"Key not found {subkey} from {key}")

            setattr(obj, keys[-1], value)
            # NOTE is it still needed?
            ## base is shared aka a bit more complicated
            # if(keys[-1] == "base"):
            #     obj.set_base(value)
            # else:
            #     setattr(obj, keys[-1], value)
