# Master Thesis DRL Planner Integration
This repository is a fork of https://github.com/ignc-research/arena-rosnav which modifies the training of drl agents by integrating the global and subgoal planner into the training pipeline. The package [arena_combined_planner_drl](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/) was created.

# Installation
Please refer to [Installation.md](docs/Installation.md) for detailed explanations about the installation process.  
  
# Configuration 
Important configuration files for the drl training itself are the following ones in the folder [default_configs](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs), those default files contain explanations of the parameters:

- **[settings.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs/settings.yaml)**: contains settings for the training process itself (environmental settings and training hyperparameters)
- **[global_planner.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs/global_planner.yaml)**: contains settings for the move_base global planner of ROS
- **[training_curriculum.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/default_agent_configs/training_curriculum.yaml)**: contains the settings of the training stages

The configuration for the robot in flatland can be found here:

- **[robot configuration](arena-rosnav/simulator_setup/robot/myrobot.model.yaml)**

For the evaluation the [run](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/run_configs/) and [taskmanager](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/task_manager_node/) configs have to be configured:

- **[default_run_configs.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/run_configs/default_run_configs.yaml)**

- **[default_task_manager_configs.yaml](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/task_manager_node/default_task_manager_configs.yaml)**

- **[data_recorder_config.yaml](arena_navigation/arena_local_planner/evaluation/arena_evaluation/01_recording/data_recorder_config.yaml)**

## Custom Global and Subgoal Planner

To create custom global planner and subgoal planner for the training, classes of the interfaces [global_planner.py](arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/scripts/global_planners/global_planner.py) or [mid_planner.py](arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/scripts/mid_planners/dummy_mid.py) need to be implemented and speciefied in the settings.yaml.




# Usage (Start a Training)

## Commandline Usage
To start a training configure first all config files mentioned in [Configuration](#configuration).

First open a terminal and activate the environment start the flatland simulation with the following commands:

```
workon rosnav
roslaunch arena_bringup start_training.launch train_mode:=true num_envs:=32 map_file:=random_map show_rviz:=false
```

Start a second terminal and start the training with the following commands:
```
workon rosnav
roscd arena_combined_planner_drl
python3 scripts/train.py -cf configs/default_agent_configs/
```

arguments for the train.py script

|long argument     | short argument| description |
|------------------|---------------|-------------|
| --configs_folder | -cf           |[string] path to config folder containing the mentioned files: *settings.yaml*, *global_planner.yaml* and *training_curriculum.yaml*|
| --eval_frequency | -ef           |[integer] overrides evaluation frequency in the settings.yaml, might be useful if another number of environments is used as expected from the settings. The evaluation frequency in total time steps is *eval_frequency* * *num_env*|
| --n_env          | -ne           |[integer] number of environments, if not set or 0, the number specified when launching flatland is used|
| --log_level      | -l            |[string] specifies the ros log level. ('debug', 'info', 'warn', 'error')|


## Docker Usage
A  [Docker file](docker/Dockerfile) is given to start a training as following:

First get the Docker file with the following command.

```
curl -LJO https://raw.githubusercontent.com/AmelusVermin/arena-rosnav/drl_combined_planner_learning/docker/Dockerfile
```
After that, build the image. All subfolders are copied into the **[workdir](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/)**. That way it is possible to add continue a training process by adding the model folder with its configs.

```
docker build -t <docker_name>/ros_training .
```

To start a training use the following example command. 

```
docker run --name train_agent_1 -p 6006:6006 --gpus all amelus/ros_training ./docker/run_training.sh -n 32 -c configs/default_agent_configs/
``` 
In the follwing tabels are important arguments of docker run command and the used .sh script.<br>

arguments for **docker run** command:
|argument| description |
|--------|-------------|
| *-p*   | port fowarding from inside the container to outside for tensorboard: \<port host machine>:\<port container> |
|*--name*| name of the container
|*--gpus*|selecting the gpus for the container, in the example all gpus|

<br>

arguments for start script inside the container: *[run_training.sh](docker/run_training.sh)*

|argument| description |
|--------|-------------|
| *-n*   |[integer] number of environments|
| *-c*   |[string] path to config folder containing the mentioned files: *settings.yaml*, *global_planner.yaml* and *training_curriculum.yaml*|
| *-e*   |[integer] overrides evaluation frequency in the settings.yaml, might be useful if another number of environments is used as expected from the settings. The evaluation frequency in total time steps is *eval_frequency* * *num_env*|


After the training the data in the specified output folder in the configs can be extracted from the container as following:
```
sudo docker cp <container name>:/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/models <folder on host machine>/       

sudo docker cp <container name>:/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/logs <folder on host machine>/       
```


# Evaluation

The evaluation in the thesis of the agents in random environments was done as following.
In the *[evaluation](arena_navigation/arena_local_planner/evaluation/arena_evaluation/)* folder can be the scripts found.

## Recording
The configs files for the evaluation mentioned in [Configuration](#configuration) were set.


First the flatland simulation need to be started. *task_file* specifies the *[task manager config file](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/task_manager_node/default_task_manager_configs.yaml)*. The following command was used as an example for the qunatitative evaluation.
```
roslaunch arena_bringup start_arena_flatland_random_maps.launch disable_scenario:="false" train_mode:=false map_file:=random_map use_recorder:=true record_only_planner:=false task_file:=outdoor_dyn_obs_05   
```

For the qualitative evaluation the following command was used as an example
```
roslaunch arena_bringup start_arena_flatland_random_maps.launch disable_scenario:="false" train_mode:=false map_file:=random_map use_recorder:=true record_only_planner:=false task_file:=indoor_dyn_obs_10_one_map
```

Then the agent need to be started. *configs* specifies the *[run config file](arena_navigation/arena_local_planner/learning_based/arena_combined_planner_drl/configs/run_configs/default_run_configs.yaml)*. For the quantitative evaluation the following command was used.
```
roslaunch arena_combined_planner_drl launch_agent.launch config:=configs/run_configs/run_agent_1.yaml
```

## Data Preparation and Plotting

First the recorded csv files need to be analyzed and combined. change to the directory of the [evaluation folder](arena_navigation/arena_local_planner/evaluation/arena_evaluation/02_evaluation/)

```
roscd arena_evaluation
cd 02_evaluation
```
Run the script to calculate metrics and join the data. The argument speciefies the folder containing the csv files from recording. It creates the *[joined_data_stats.csv](arena_navigation/arena_local_planner/evaluation/arena_evaluation/02_evaluation/joined_data_stats.csv)* file.
```
python3 analyze_and_join_data.py marvin_quantitative_data/
```

The plots were created with the *[plot_data.py](arena_navigation/arena_local_planner/evaluation/arena_evaluation/03_plotting/plot_data.py)* script.
For this script the following files in *[03_plotting](arena_navigation/arena_local_planner/evaluation/arena_evaluation/03_plotting/)* are assumed. 
- a *[folder](arena_navigation/arena_local_planner/evaluation/arena_evaluation/03_plotting/marvin_data_qualitative/)* containing the csv files of the qualitative evaluation recordings.
- a *[joined_data_stats.csv](arena_navigation/arena_local_planner/evaluation/arena_evaluation/03_plotting/joined_data_stats.csv)* file   
- a *[full_stats_quantitative.csv](arena_navigation/arena_local_planner/evaluation/arena_evaluation/03_plotting/full_stats_quantitative.csv)* file, containing statistics for comparison of different planners. 

The plots are saved in the *[plots](arena_navigation/arena_local_planner/evaluation/arena_evaluation/03_plotting/plots/)* subfolder
```
roscd arena_evaluation
cd 03_plotting
python3 plot_data.py
```



# Used third party repos:
* Flatland: http://flatland-simulator.readthedocs.io
* ROS navigation stack: http://wiki.ros.org/navigation
* Pedsim: https://github.com/srl-freiburg/pedsim_ros
* Stable Baselines (Tensorflow): https://stable-baselines.readthedocs.io/en/master/index.html
