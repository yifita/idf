import sys
import matplotlib as mpl
import torch
import numpy as np
mpl.use('Agg')
from config import ConfigObject, build_config_file
import importlib
from typing import List,Dict,Iterable,Set
import os
import json
from collections import defaultdict
import time

from helper import AllOf

typenames = {
    "trainer" : "trainer.",
    "network" : "network.",
    "loss" : "loss.",
    "data" : "data.",
    "evaluator" : "evaluator.",
    "logger" : "logger.",
    "tasks" : "task."

}

def mkdir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)


class Runner(ConfigObject):

    def __init__(self, config):
        # unused in runner class see executor for usage
        self.depends : List[str] = []
        self.skip_if_exists : str = None
        self.name : str = "run"
        self.epochs : int = 1000
        self.learning_rate : float = 1e-4
        self.tasks : List["Task"] = []
        self.active_task : "Task" = None
        # self.network = object
        # self.loss = object
        # self.data = object
        name = config.get("name", self.name)
        self.folder = os.path.join("runs", name)
        mkdir(self.folder)
        self.evaluator : "Evaluator" = None
        self.data_path : str = "./"
        self.name = name
        self.perf_test = False

        super().__init__(config)
        if(self.perf_test):
            self.evaluators =  AllOf([])
            self.loggers = AllOf([])
            import logging
            logging.disable(logging.CRITICAL)


    @staticmethod
    def filterdict(config : Dict[str,object]):
        config.pop("runner",None)
        stack : List[Iterable] = [config.values()]
        while len(stack) > 0:
            item = stack.pop()

            for sub_item in item:
                if(isinstance(sub_item,dict)):
                    sub_item.pop("runner",None)
                    stack.append(sub_item.values())
                if isinstance(sub_item,list):
                    stack.append(sub_item)

        return config

    @staticmethod
    def get_config_file(suffix:str, name:str):
        return f".config/{name}_{suffix}.json"

    @staticmethod
    def savehash(suffix: str, config : Dict[str,object]):
        config = Runner.filterdict(config.copy())
        if not os.path.exists(f".config/"):
            os.mkdir(".config")
        filename = Runner.get_config_file(suffix,config["name"])
        file = open(filename,"w")
        file.write(json.dumps(config,skipkeys=True,sort_keys=True))
        file.flush()
        file.close()
        return filename

    @staticmethod
    def comparehash(suffix:str, config : Dict[str,object]) -> bool:
        config = Runner.filterdict(config.copy())
        path = Runner.get_config_file(suffix,config["name"])
        if not os.path.exists(path):
            return False
        file = open(path,"r")
        jsonText = file.read()
        file.close()
        return json.dumps(config,skipkeys=True,sort_keys=True) == jsonText

    @classmethod
    def get_class_for(self, name:str, classname: str):
        basename = typenames[name]
        basename += classname.lower()
        return getattr(importlib.import_module(basename),classname)

    def run(self):

        if(self.perf_test):
                print("Starting Script in testing mode all logging disabled")

        if(self.skip_if_exists is not None and os.path.exists(self.skip_if_exists)):
            return
        if(self.comparehash("runner",self.key_value)):
            self.py_logger.warn(f"Skipping runner {self.name} because config has not changed since last run\n")

        #remove runfile
        runfile = Runner.get_config_file("runner",self.name)

        if(os.path.exists(runfile)):
            os.remove(runfile)

        self.py_logger.info(f"Running runner {self.name}\n")
        current_time = time.time()

        for task in self.tasks:
            self.active_task = task
            task()

        self.py_logger.info(f"Runner {self.name} has finished\n")
        self.savehash("runner",self.key_value)
        if(self.perf_test):
            print(f"Script took {time.time()-current_time}s to compute")


from copy import copy

class argdict(defaultdict):
    #this is a static variable as it is never changed etc
    args : Set[str] = set()
    def __init__(self):
        super().__init__(str)

    def __missing__(self, key):
        self.args.add(key)
        return super().__missing__(key)

    def copy(self):
        d = argdict()
        d.update(self)
        return d


def parse_json_arg(arg:str) -> object:
    if(arg == "null"):
        return None
    if(arg=="true"):
        return True
    if(arg == "false"):
        return False
    if(arg.isnumeric()):
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

if __name__ == "__main__":
    import sys
    if(len(sys.argv) > 1):
        path = sys.argv[1]
    else:
        path = os.path.dirname(__file__) + "/../example.json"

    argDict = argdict()
    force_usage = True

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if("--help" not in sys.argv):
        force_usage = False
        currentKey = None
        for arg in sys.argv:
            if(arg.startswith("--")):
                if(currentKey is not None):
                    print(f"stray key {currentKey} ignored!")
                currentKey = arg[2:]

            elif(currentKey is not None):
                argDict[currentKey] = parse_json_arg(arg)
                currentKey = None
            else:
                print(f"stray value {arg} ignored!")

        if(currentKey is not None):
            print(f"stray key {currentKey} ignored!")

    print("\nParsed Arguments:")
    print(argDict.items())
    print("")
    folder = os.path.dirname(path)
    file = path[len(folder):]
    json_obj = build_config_file(file,folder,argDict)


    if(force_usage or len(argdict.args)>0):
        folder = os.path.dirname(path)
        file = path[len(folder):]
        json_obj = build_config_file(file, folder, argDict)
        if( not force_usage):
            print("Not all arguments defined, missing arguments:")
        else:
            print("Usage\n")
            print("Help for script:")
            print(f"{sys.argv[1]}\n")
        for k in argDict.args :
            print(f"--{k} <{k}>")
        print("")
        print("All arguments must be set no defaults! ")
        print("")

        sys.exit(0)

    if(isinstance(json_obj,list)):
        print("")
        print(f"script {path} is an array, try running it with executer.py!")
        print("")
        sys.exit(0)

    ##stop it before it creates any additional files we want to be able to rerun
    for d in json_obj.get("depends",[]):
        runfile = Runner.get_config_file("runner", d)
        if not os.path.exists(runfile):
            print(f"Runner {json_obj['name']} can't run because job {d} did not finish", file=sys.stderr)
            sys.exit(-1)

    runner = Runner(json_obj)
    runner.run()
    sys.exit(0)
