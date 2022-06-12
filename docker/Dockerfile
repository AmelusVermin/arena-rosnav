# start from an Ubuntu 18.04 image with installed CUDA 10.0 and cudnn7
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# set workdir
SHELL ["/bin/bash", "-c"]

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# update all programs and software properties common for adding repos
RUN apt-get update \
&&  apt-get -y upgrade \
&&  apt-get -y install software-properties-common

# add repositories for ROS
RUN add-apt-repository universe \
&&  add-apt-repository multiverse \
&&  add-apt-repository restricted \
&&  apt-get update

# setup source.list and keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
&&  apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
&&  apt-get update

# set time zone
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt-get -y install tzdata

# install ROS
RUN apt-get install -y --no-install-recommends ros-melodic-desktop-full 

# setup environment

RUN echo $"source /opt/ros/melodic/setup.bash" >> /root/.bashrc \
&&  source ~/.bashrc

# install some ROS packages
RUN apt-get install -y \
        python-rosdep \ 
        python-rosinstall \
        python-rosinstall-generator \ 
        python-wstool \
        build-essential

# initialize ROS
RUN rosdep init \
&&  rosdep update

# clone repo and setup arena-rosnav 
RUN mkdir -p catkin_ws/src \
&&  cd catkin_ws/src \
&&  git clone -b drl_combined_planner_learning https://github.com/AmelusVermin/arena-rosnav.git

RUN mkdir -p geometry2_ws/src \
&&  cd geometry2_ws/src \
&& git clone -b melodic-devel https://github.com/ros/geometry2.git

# install further packages
RUN apt-get update \
&&  apt-get install -y \
        libqt4-dev \
        libopencv-dev \
        liblua5.2-dev \
        screen \
        python3 \
        python3-pip \
        python3.6-dev \
        libpython3.6-dev \
        python3-catkin-pkg-modules \
        python3-rospkg-modules \
        python3-empy \
        python3-setuptools \
        ros-melodic-navigation \
        ros-melodic-teb-local-planner \
        ros-melodic-mpc-local-planner \
        libarmadillo-dev \
        ros-melodic-nlopt \
        nano \
        tmux \
        htop

# update pip
RUN pip3 install --upgrade pip
# install python packages
RUN pip3 install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag tf tf2_ros --ignore-installed \ 
&&  pip3 install pyyaml \
        rospkg \
        scikit-learn \
        seaborn \
        catkin_pkg \ 
        netifaces \ 
        pathlib \
        filelock \
        PyQt5 \
        cloudpickle==1.6.0 \
        joblib==1.0.1 \
        matplotlib==3.3.4 \
        opencv-python==4.5.1.48 \
        pandas==1.1.5 \
        scipy==1.5.4 \
        gym==0.18.0 \
        scikit-image==0.17.2 \
        tensorflow-gpu==1.15 \
        stable-baselines==2.10.2 \
        imageio==2.9.0 \ 
        Pillow==7.2.0 \
        Pympler 
        


# load forks and build project
RUN cd catkin_ws/src/arena-rosnav \
&&  rosws update \
&&  rosdep update 

RUN source /opt/ros/melodic/setup.bash \
&&  source /root/.bashrc \
&&  cd geometry2_ws \
&&  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 \ 
&&  source devel/setup.bash

RUN source /opt/ros/melodic/setup.bash \
&&  source /root/.bashrc \
&&  cd catkin_ws \
&&  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 \ 
&&  source devel/setup.bash

RUN sed -i '/opt\/ros\/melodic\/setup.bash/i export PYTHONPATH=""' /root/.bashrc 
RUN echo $'export PYTHONPATH=/geometry2_ws/devel/lib/python3/dist-packages:${PYTHONPATH}\n' >> /root/.bashrc 
RUN echo $'export PYTHONPATH=/catkin_ws/src/arena-rosnav:${PYTHONPATH}\n' >> /root/.bashrc 
RUN echo $'source /catkin_ws/devel/setup.bash\n' >> /root/.bashrc

WORKDIR /catkin_ws/src/arena-rosnav/
COPY . .