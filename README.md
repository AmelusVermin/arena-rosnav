# Docker

```
curl -LJO https://raw.githubusercontent.com/AmelusVermin/arena-rosnav/drl_combined_planner_learning/docker/Dockerfile
docker build -t amelus/ros_training .
docker run --name train_agent_1 -p 6006:6006 --gpus all amelus/ros_training ./docker/run_training.sh -n 16 -c configs/configs_agent_1/
``` 

# Master Thesis DRL Planner Integration
This repository is a fork of https://github.com/ignc-research/arena-rosnav which modifies the training of drl agents by integrating the global and subgoal planner into the training pipeline.

# Installation
Please refer to [Installation.md](docs/Installation.md) for detailed explanations about the installation process.  
  
# Configuration
Important configuration files for the drl training itself are the following ones in the folder [default_configs](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs), those default files contain explanations of the parameters:

- **[settings.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs/settings.yaml)**: contains settings for the training process itself (environmental settings and training hyperparameters)
- **[global_planner.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs/global_planner.yaml)**: contains settings for the move_base global planner of ROS
- **[training_curriculum.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs/training_curriculum.yaml)**: contains the settings of the training stages

The configuration for the robot in flatland can be found here::

- **[robot configuration](arena-rosnav/simulator_setup/robot/myrobot.model.yaml)**

For the evaluation the [run](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/run_configs/) and [taskmanager](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/task_manager_node/) configs have to be configured:

- **[default_run_configs.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/run_configs/default_run_configs.yaml)**

- **[default_task_manager_configs.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/task_manager_node/default_task_manager_configs.yaml)**


# Usage

## Commandline Usage


## Docker Usage
A  Docker file is given to start a training as following:

```
curl -LJO https://raw.githubusercontent.com/AmelusVermin/arena-rosnav/drl_combined_planner_learning/docker/Dockerfile
docker build -t <docker_name>/ros_training .
docker run --name train_agent_1 -p 6006:6006 --gpus all amelus/ros_training ./docker/run_training.sh -n 16 -c configs/configs_agent_1/
``` 



# Evaluation

## Requirements
For the evaluation another Python environment with Python 3.7 is necessary.

```
mkvirtualenv --python=python3.7 rosnav-eval
workon rosnav-eval
```

Furthermore, some packages need to be insalled.

```
pip install 
```



# Used third party repos:
* Flatland: http://flatland-simulator.readthedocs.io
* ROS navigation stack: http://wiki.ros.org/navigation
* Pedsim: https://github.com/srl-freiburg/pedsim_ros
