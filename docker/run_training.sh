#!/bin/bash

while getopts "n:c:" opt
do
    case "${opt}" in
        n ) num_env="$OPTARG" ;;
        c ) config_path="$OPTARG" ;;
        esac
done

source $HOME/.bashrc
source $HOME/catkin_ws/devel/setup.bash 
source $HOME/.profile
source `which virtualenvwrapper.sh`

roslaunch arena_bringup start_training_2.launch train_mode:=true num_envs:=$num_env map_file:=random_map show_rviz:=false &>/dev/null &
sleep 5

workon rosnav
roscd arena_combined_planner_drl/ 
python3 scripts/train.py -cf $config_path