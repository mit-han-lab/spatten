import os
import shutil

class SpAtten:
    def __init__(self, workdir=""):
        if workdir:
            self.workdir = workdir
        else:
            self.workdir = os.path.join(os.curdir, "spatten.workdir")
        self.__check()
        self.__compile()

    def simulate(self, taskfile, num_bufline, topk_parallelism, num_multipliers, bandwidth_downsample, select_iteration, with_waveform):
        options = ""
        if with_waveform:
            options += " -Dspatten.withWaveform=1 "
        if select_iteration >= 0:
            options += f" -Dspatten.selectIteration={select_iteration} "
        ret = os.system(f"cd '{self.workdir}'; env LD_LIBRARY_PATH='{os.path.abspath(self.workdir)}/dpi':'{os.path.abspath(self.workdir)}/third_party/ramulator2' java {options} -cp spatten-assembly-1.0.jar spatten.sim.TestSpAtten '{os.path.abspath(taskfile)}' '{num_bufline}' '{topk_parallelism}' '{num_multipliers}' '{bandwidth_downsample}'")
        if ret != 0:
            raise RuntimeError(f"Failed to run the simulation, return value={ret}")
        

    def __check(self):
        dependencies = ["java", "sbt", "verilator"]
        for dep in dependencies:
            if shutil.which(dep) is None:
                raise RuntimeError(f"Dependency {dep} is not available")

    def __compile(self):
        os.makedirs(self.workdir, exist_ok=True)

        hardware_dir = os.path.join(os.path.dirname(__file__), "spatten_hardware/hardware")
        jar_path = os.path.join(hardware_dir, "target/scala-2.13/spatten-assembly-1.0.jar")
        extra_files = ["ramulator_config.yaml", "dpi", "third_party"]

        ret = os.system(f"cd '{hardware_dir}'; sbt assembly")
        if ret != 0:
            raise RuntimeError(f"Failed to compile SpAtten source code, return value={ret}")
        if not os.path.exists(jar_path):
            raise RuntimeError(f"Cannot find compiled jar file")
        
        shutil.copy(jar_path, self.workdir)
        for f in extra_files:
            os.system(f"ln -sf '{os.path.join(hardware_dir, f)}' '{os.path.join(self.workdir, f)}'")
            # os.symlink(os.path.join(hardware_dir, f), os.path.join(self.workdir, f))
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("taskfile", type=str)
    parser.add_argument("--num_bufline", type=int, default=2048)
    parser.add_argument("--topk_parallelism", type=int, default=16)
    parser.add_argument("--num_multipliers", type=int, default=64)
    parser.add_argument("--bandwidth_downsample", type=int, default=8)
    parser.add_argument("--select_iteration", type=int, default=-1)
    parser.add_argument("--with_waveform", action="store_true")
    args = parser.parse_args()

    spatten = SpAtten()
    spatten.simulate(args.taskfile, args.num_bufline, args.topk_parallelism, args.num_multipliers, args.bandwidth_downsample, args.select_iteration, args.with_waveform)