#!/bin/bash

while getopts "n:c:" opt
do
    case "${opt}" in
        n ) num_env="$OPTARG" ;;
        c ) config_path="$OPTARG" ;;
        esac
done

git pull
cd /
source /root/.bashrc
source catkin_ws/devel/setup.bash 

roslaunch arena_bringup start_training_2.launch train_mode:=true num_envs:=$num_env map_file:=random_map show_rviz:=false &>/dev/null &
sleep 5

roscd arena_combined_planner_drl/ 
python3 scripts/train.py -cf $config_path