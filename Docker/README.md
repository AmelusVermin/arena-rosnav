## Install docker
https://docs.docker.com/docker-for-mac/install/

## Build image and setup container
1. clone the whole repositary
```
git clone https://github.com/Herrsun/arena-rosnav
```
2. Open local terminal under the path of arena-rosnav
3. Run `docker-compose up --build`

## Start /Enter /Stop the container
1. Check if the arena-rosnav and novnc container is already started
```
docker ps
```
2. Start container (under the path of arena-rosnav)
```
docker-compose start
```
3. Enter the ros contaier 
```
docker exec -it arena-rosnav bash
```
4. Stop the container (under the path of arena-rosnav)
```
docker-compose stop 
```

## Show the GUI app in ROS by browser
1. Open 'http://localhost:8080/vnc.html'
2. You can adjust the window size to automatic local scale or set fullscreen though the side bar on the left

## Test
1. After entering the container by `docker exec -it arena-rosnav bash` activate the venv by
```
workon rosnav
```
2. Roslaunch in venv
```
roslaunch arena_bringup start_arena_flatland.launch train_mode:=false use_viz:=true local_planner:=dwa map_file:=map1 obs_vel:=0.3

```
3. Open http://localhost:8080/vnc.html you will see the rviz window in browser
3. By setting a 2D Nav Goal manually on white goal point you will see the robot move to the goal automatically

## Develop in VS Code
1. Install plugin Remote - Containers in VS Code
2. Click the green icon in the lower left corner (or type `F1`) 
3. Select `Remote-Containers: Reopen in Container`
4. Select `From docker-compose.yml`
5. Select `ros`
