from utils.argparser import get_commandline_arguments, get_config_arguments, check_params
from utils.startup_utils import wait_for_nodes, make_envs
from pydoc import locate
from pympler import muppy, summary, tracker, refbrowser
from utils.hyperparameter_utils import write_hyperparameters_json
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv
import pickle
import os
import rospkg
import rospy
import cProfile
import pstats
ENV = None
def sim():
    print("loop")
    for i in range(20):
        print("reset loop")
        ENV.reset()
        for j in range(1000):
            _, _, done, info = ENV.step([(0,0)])
            if done:
                print("reset done")
                ENV.reset()
    print("loop end")

def main():

    args = get_commandline_arguments()
    # (environment_settings.yaml from this package and myrobot.model.yaml from simulator_setup)
    if args.configs_folder is "default":
        args.configs_folder = os.path.join(
            rospkg.RosPack().get_path("arena_combined_planner_drl"), 
            "configs",
            "configs"
        )

    settings_file = os.path.join(
            args.configs_folder, 
            "settings.yaml"
        )
    
    args, save_paths = get_config_arguments(args, settings_file)

    # initiate ros node with according log level
    rospy.set_param("/curr_stage", args.task_curr_stage)
    args.n_envs = 1
    #check_params(args)
    args.n_steps = int(args.batch_size / args.n_envs)

    # wait for nodes to start
    wait_for_nodes(with_ns=True, n_envs=args.n_envs, timeout=5)

    # get classes of global and mid planner
    global_planner = locate(args.global_planner_class)
    mid_planner = locate(args.mid_planner_class)
    write_hyperparameters_json(args, save_paths)
    global ENV
    print("start init")
    ENV = DummyVecEnv([make_envs(args, True, 0, global_planner, mid_planner, save_paths, train=True)])

    ENV = VecNormalize(
                ENV, training=True, norm_obs=True, norm_reward=False, clip_reward=15
            )

    ENV.reset()
    ENV.step([(0,0)])
    

    #for i in range(20000):
    #    env.envs[0]._call_service_takeSimStep(env.envs[0]._action_frequency)

    
    all_objects_before = muppy.get_objects()
    cProfile.run('sim()', 'pr')
    all_objects_after = muppy.get_objects()
    
    stats = pstats.Stats('pr')
    stats.sort_stats("time")

    sum1 = summary.summarize(all_objects_before)
    #sum2 = summary.summarize(all_objects_between)
    sum3 = summary.summarize(all_objects_after)
    #ib = refbrowser.ConsoleBrowser(env.envs[0], maxdepth=4)
    with open("mem_stats.pkl", "wb") as f:
        pickle.dump((sum1, sum3), f)
    #diff1 = summary.get_diff(sum1, sum2)
    diff2 = summary.get_diff(sum1, sum3)

    #summary.print_(diff1)
    print()
    summary.print_(diff2)

    stats.print_stats()
    stats.dump_stats(filename='stats3.prof')
    
    #ib.print_tree()

if __name__ == "__main__":
    main()