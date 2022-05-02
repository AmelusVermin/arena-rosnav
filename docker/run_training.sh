#!/bin/bash

while getopts "n:c:e:" opt
do
    case "${opt}" in
        n ) num_env="$OPTARG" ;;
        c ) config_path="$OPTARG" ;;
        e ) eval_freq="$OPTARG" ;;
        esac
done

git pull
cd /
source /root/.bashrc

export PYTHONPATH=""
source opt/ros/melodic/setup.bash
export PYTHONPATH=/geometry2_ws/devel/lib/python3/dist-packages:${PYTHONPATH}
export PYTHONPATH=/catkin_ws/src/arena-rosnav:${PYTHONPATH}
source catkin_ws/devel/setup.bash

roslaunch arena_bringup start_training_2.launch train_mode:=true num_envs:=$num_env map_file:=random_map show_rviz:=false #&>/dev/null &
sleep 10

roscd arena_combined_planner_drl/ 
#if [ -z ${eval_freq+x} ]; 
#then 
    #echo "start training with eval frequency given in config file"
    #python3 scripts/train.py -cf $config_path 2>&1 | tee output.txt
#else 
    #echo "start training with eval frequency from command line"
    #python3 scripts/train.py -cf $config_path -ef $eval_freq 2>&1 | tee output.txt
#fi
