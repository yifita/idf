from runner import Runner,argdict,parse_json_arg
from typing import List,Dict

from config import build_config_file
import sys
import os
from dataclasses import dataclass

import subprocess
import shutil


@dataclass
class Job(object):
    name : str = ""
    command : str = ""
    depends : List[str] = None
    config : Dict[str,object] = None
    index :int = -1





#we are running locally on one GPU
if(shutil.which("bsub") is None):

    def kill_job(name:str, dry:bool):
        return False

    def execute_job(job:Job):
        name = job.name
        command = job.command
        depends = job.depends
        print(f"Running Runner {name}\n")
        p = subprocess.run(command, shell=True, check=True)
        print(f"{name},{command},{depends}\n")
        print(f"Finished Runner {name}\n")

#cluster time
else:
    #arguments
    kwargs = {"R":"rusage[mem=40GB,scratch=10000,ngpus_excl_p=1]","n":"4","W":"4:00"}
    jobs = {}
    def kill_job(name:str, dry : bool):
        global jobs
        p =subprocess.check_output(f"bjobs -J \"{name}\"  -o \"jobid\"", shell=True, stderr=subprocess.DEVNULL).decode()
        if(len(p) > 5):
            tokens = p.splitlines()
            job_id = tokens[1]
            print(f"Already running runner {name} with jobid {job_id}")
            if not dry:
                print(f"Killing {name}")
                subprocess.run(f"bkill {job_id}")
                return False
            return True
        return False

    def execute_job(job:Job):
        global kwargs
        global jobs
        name = job.name
        command = job.command
        depends = job.depends
        print(f"{name} {command} {depends} ")
        args = kwargs.copy()
        args["J"] = name
        p =subprocess.check_output(f"bjobs -J \"{name}\"  -o \"jobid\"", shell=True, stderr=subprocess.DEVNULL).decode()
        if(len(p) > 5):
            tokens = p.splitlines()
            job_id = tokens[1]
            jobs[name]=job_id
            print(f"Already running runner {name} with jobid {jobs[name]}")
            print(f"There was a config chance ")
            return

        if(len(depends)> 0):
            warg = ""
            for d in depends:
                job_id = jobs.get(d,None)
                if(job_id is not None):
                    warg += f"ended({job_id})&&"
            warg = warg[:-2]
            args["w"]=warg

        bcommand = "bsub "
        for k,v in args.items():
            bcommand+= f"-{k} \"{v}\" "
        bcommand += command
        print( bcommand)
        p = subprocess.check_output(bcommand, shell=True).decode()
        start = p.index('<')+1
        end = p.index('>',start)
        jobs[name]=p[start:end]
        print(f"Executed runner {name} with jobid {jobs[name]}")


class Scheduler(object):

    runner_path = os.path.dirname(__file__) + "/runner.py"
    no_job = Job()

    jobs:Dict[str,Job]
    index:int

    def __init__(self):
        self.jobs = {}
        self.scheduled = {}
        self.index = 0


    def check_if_job_is_runnable_and_kill(self, job:Job) -> bool:
        name = job.name
        if(len(job.depends) > 0):
            #print(f"Job {name} dependcies {job.depends} is running => job is rerun")
            kill_job(name,False)
            return True

        if(Runner.comparehash("runner",job.config)):
            #print(f"Job {name} config has not changed")
            return False
        #only kill job in case the config has changed
        if(kill_job(name,Runner.comparehash("exec",job.config))):
            #print(f"Job {name} will continue running")
            self.jobs[name] = Scheduler.no_job
            return False

        return True


    def schedule_maybe(self, job:Dict[str,object]):
        name = job["name"]
        if(name in self.jobs):
            #print(f"Job {name} already scheduled \n")
            return
        command = f"{sys.executable} \"{Scheduler.runner_path}\" "
        jobO = Job(name,command,job.get("depends",[]).copy(),job)
        self.jobs[name] = jobO
        self.schedule_if_possible(jobO)

    def schedule_if_possible(self, job:Job):
        for d in job.depends:
            dependedjob = self.jobs.get(d,None)
            if(dependedjob is None or dependedjob.index < 0) and dependedjob != Scheduler.no_job :
                return False

        for d in job.depends.copy():
            if(self.jobs.get(d,None) == Scheduler.no_job):
                job.depends.remove(d)

        if(self.check_if_job_is_runnable_and_kill(job)):
            file = Runner.savehash("exec",job.config)
            job.command += file
            job.index = self.index
            #print(f"{self.index} : {job.name}")
            self.index += 1
        else:
            self.jobs[job.name] = Scheduler.no_job

        for sjob in self.jobs.values():
            if(sjob == Scheduler.no_job):
                continue
            if (sjob.index < 0) and (job.name in sjob.depends):
                self.schedule_if_possible(sjob)


        return True

    def run_jobs_for(self, targets:List[str]) -> List[Job]:
        jobs :List[Job] = []

        if(len(targets)==0):
            jobs = list(filter(lambda x:x != Scheduler.no_job,self.jobs.values()))
        else:
            frontier = targets
            jobs :List[Job] = []
            while(len(frontier) > 0):
                jobName = frontier.pop()
                job = self.jobs[jobName]
                if job != Scheduler.no_job:
                    frontier += job.depends
                    jobs.append(job)
            #print(jobs)
        jobs.sort(key=lambda x:x.index)
        for job in jobs  :
            execute_job(job)



    def print_graph_python(self):
        index = dict()
        maxLevel = 0
        for k,v in sorted(self.jobs.items(),key=lambda x:x[1].index):
            if(v == Scheduler.no_job):
                index[k] = -1
            elif(len(v.depends) == 0):
                index[k] = 0
            else:
                level = 0
                for d in v.depends:
                    level = max(index[d]+1,level)
                index[k] = level
                maxLevel = max(level,maxLevel)
        current=-2

        for k,v in sorted(index.items(),key=lambda x:x[1]):
            if(current<v):
                if(v==-1):
                    print("\nSkipped:")
                else:
                    print(f"\nLevel {v}:")
                current=v
            print(f"\t{k}")


def execute_on_cluster(path : str, targets :List[str] = [], args:List[str] = []):

    argDict = argdict()
    force_usage = True

    #parse args
    if("--help" not in args):
        force_usage = False
        for arg in args:
            name,value = arg.split("=", 1)
            argDict[name[2:]] = parse_json_arg(value)

    print("\nParsed Arguments:")
    print(argDict.items())
    print("")
    folder = os.path.dirname(path)
    file = path[len(folder):]
    json_obj = build_config_file(file,folder,argDict)

    if(force_usage or len(argdict.args)>0):
        folder = os.path.dirname(path)
        file = path[len(folder):]
        json_obj = build_config_file(file,folder,argDict)
        if( not force_usage):
            print("Not all arguments defined, missing arguments:")
        else:
            print("Usage\n")
            print("Help for script:")
            print(f"{sys.argv[1]}\n")
        for k in argDict.args :
            print(f"--{k}=<{k}>")
        print("")
        print("All arguments must be set no defaults! ")
        print("")

        sys.exit(0)

    print(f"using targets = {targets}")

    if not isinstance(json_obj,list):
        print("Executor can only work on a list of runners")
        return

    sh = Scheduler()

    for r in json_obj:
        sh.schedule_maybe(r)

    if(len(targets) > 0 and targets[0] == "print"):
        sh.print_graph_python()
    else:
        sh.run_jobs_for(targets)

if __name__ == "__main__":

    if(len(sys.argv) > 1):
        path = sys.argv[1]
        targets = []
        args = []
        if(len(sys.argv) > 2):
            targets = [x for x in sys.argv[2:] if not x.startswith("--")]
            args = [x for x in sys.argv if x.startswith("--") ]
        execute_on_cluster(path,targets,args)
    else:
        print(f"Usage {sys.argv[0]} json <target target target>")

