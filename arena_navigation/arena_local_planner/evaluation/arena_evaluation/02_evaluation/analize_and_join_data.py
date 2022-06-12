import pandas as pd
import os
import numpy as np
import glob
import warnings
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="folder containing csv files from records")
    return parser.parse_args()

def get_AOL(yaws, path_length):
    """ calculate angle over length"""
    AOL_list = []
    cusps_list = []

    total_yaw = 0
    cusps = 0
    for i, yaw in enumerate(yaws):
        if i == 0:
            continue
        else:
            yaw_diff = np.abs(yaw-yaws[i-1])
            if yaw_diff == np.pi:
                cusps += 1
            total_yaw += yaw_diff
    if path_length != 0:
        AOL_list.append(total_yaw/path_length)
        cusps_list.append(cusps)
    else:
        AOL_list.append(np.nan)
        cusps_list.append(np.nan)
    return AOL_list, cusps_list

def get_curvature(pos_x, pos_y):
    """ calculate menger curvature for path"""
    curvature_list = []
    normalized_curvature_list = []
    
    points = [list(x) for x in zip(pos_x, pos_y)]
    for i, point in enumerate(points):
        try:
            x = np.array(point)
            y = np.array(points[i+1])
            z = np.array(points[i+2])
            curvature_list.append(calc_curvature(x, y, z)[0])
            normalized_curvature_list.append(
                calc_curvature(x, y, z)[1])
            continue
        except:
            curvature_list.append(np.nan)
            normalized_curvature_list.append(np.nan)
            continue
    return curvature_list, normalized_curvature_list

def calc_curvature(x, y, z):  # Menger curvature of 3 points
    """ calculate menger curvature for single set of 3 points """
    triangle_area = 0.5 * \
        np.abs(x[0]*(y[1]-z[1]) + y[0]*(z[1]-x[1]) + z[0]*(x[1]-y[1]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        curvature = 4*triangle_area / \
            (np.abs(np.linalg.norm(x-y)) * np.abs(np.linalg.norm(y-z))
                * np.abs(np.linalg.norm(z-x)))
        normalized_curvature = curvature * \
            (np.abs(np.linalg.norm(x-y)) + np.abs(np.linalg.norm(y-z)))
    return [curvature, normalized_curvature]

def get_roughness(pos_x, pos_y):
    """ calculate roughness for whole path """
    roughness_list = []
    points = [list(x) for x in zip(pos_x, pos_y)]
    
    for i, point in enumerate(points):
        try:
            x = np.array(point)
            y = np.array(points[i+1])
            z = np.array(points[i+2])
            roughness_list.append(calc_roughness(x, y, z))
            continue
        except:
            roughness_list.append(np.nan)
            continue
    return roughness_list

def calc_roughness(x, y, z):
    """ calculate roughness for single set of 3 points """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        triangle_area = 0.5 * \
            np.abs(x[0]*(y[1]-z[1]) + y[0]*(z[1]-x[1]) + z[0]*(x[1]-y[1]))
        # basically height / base (relative height)
        roughness = 2 * triangle_area / np.abs(np.linalg.norm(z-x))**2
    return roughness

def get_jerk(lin_vel_x, lin_vel_y):
    """ calculate jerk, accelaration and velocities based on linear velocities """

    jerk_list, acc_list, vel_list = [], [], []
    velocities = [list(x) for x in zip(lin_vel_x, lin_vel_y)]

    for i, vel in enumerate(velocities):
        try:
            v1 = np.array(vel)
            v2 = np.array(velocities[i+1])
            v3 = np.array(velocities[i+2])
            jerk, acc = calc_jerk(v1, v2, v3)
            jerk_list.append(jerk)
            acc_list.append(acc)
            continue
        except:
            jerk_list.append(np.nan)
            continue
    for i, j in velocities:
        vel_list.append((i**2+j**2)**(1/2))
    for _ in range(len(jerk_list)-len(acc_list)):
        acc_list.append(None)
    return jerk_list, acc_list, vel_list

def calc_jerk(v1, v2, v3):
    """ calculate jerk and accelaration of 3 linear velocities """
    v1 = (v1[0]**2 + v1[1]**2)**0.5  # total velocity
    v2 = (v2[0]**2 + v2[1]**2)**0.5
    v3 = (v3[0]**2 + v3[1]**2)**0.5
    a1 = v2-v1  # acceleration
    a2 = v3-v2
    jerk = np.abs(a2-a1)
    acc = a1
    return jerk, acc

def get_point_distance(p1, p2):
    """ calculate distance of 2 points """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def get_path_length(path_x, path_y):
    """ calculate path length """
    sum = 0
    for i in range(len(path_x)-1):
        sum += get_point_distance([path_x[i], path_y[i]], [path_x[i+1], path_y[i+1]])
    return sum

def import_data(files):
    names = ["agent",
            "map type",
            "number obstacles",
            "episode",
            "episode time",
            "success",
            "done reason",
            "collisions",
            "path length",
            "computation time full [in ms]",
            "computation time minimal [in ms]",
            "jerk",
            "acceleration",
            "velocities",
            "roughness",
            "curvature",
            "normalized",
            "max curvature",
            "angle over length"
            ]
    data = pd.DataFrame(columns=names)

    for file in files:
        print(f"import: {file}")
        file_split = file.split("/")[-1].split("_")
        agent_name = f"Agent {file_split[1]}"
        map_type = file_split[2]
        num_obstacles = int(file_split[3].split(".")[0])
        

        df = pd.read_csv(file)
        #n = df.shape[0]

        # agent_col = [agent_name] * n
        # map_col = [map_type] * n
        # obs_col = [num_obstacles] * n

        # # prepare data frame
        # df["agent"] = agent_col
        # df["map type"] = map_col
        # df["number obstacles"] = obs_col
        # df = df.drop(["model"], axis=1)
        df = df.iloc[1:]
        df = df[df["episode"] < 150]

        for i in range(0,150):
            episode_data = df[df["episode"] == i]
            last_row = episode_data.iloc[-1]

            episode_time =last_row["ep_time"]

            
            if last_row["reached_goal"] and last_row["num_collisions"] == 0:
                success = 1
                done_reason = "goal reached"
            elif last_row["num_collisions"] > 0:
                success = 0
                done_reason = "collision"
            else:
                success =0
                done_reason = "time out"
            
            collsions = last_row["num_collisions"]
            comp_time_full = last_row["resp_time"]
            comp_time_min = last_row["resp_time_min"]
            # get path length
            path_length = get_path_length(episode_data["robot_pos_x"].values.tolist(), episode_data["robot_pos_y"].values.tolist())
            
            # get jerks, accelerations and velocities
            jerk, acc, vel = get_jerk(episode_data["robot_lin_vel_x"].values.tolist(), episode_data["robot_lin_vel_y"].values.tolist())
            
            jerk = jerk[:-2]
            accelerations = acc[:-2]
            velocities = vel
            # get roughness
            roughness = get_roughness(episode_data["robot_pos_x"].values.tolist(), episode_data["robot_pos_y"].values.tolist())
            roughness = roughness[:-2]
            # get curvature
            curv, norm_curv = get_curvature(episode_data["robot_pos_x"].values.tolist(), episode_data["robot_pos_y"].values.tolist())
            curvature = curv[:-2]
            normalized_curvature = norm_curv[:-2]
            max_curvature = max(curv)
            # get aol
            aol, _ = get_AOL(episode_data["robot_orientation"].values.tolist(), path_length)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                stats = {
                    "agent": agent_name,
                    "map type": map_type,
                    "number obstacles": num_obstacles,
                    "episode": i,
                    "episode time": episode_time,
                    "success": success,
                    "done reason": done_reason,
                    "collisions": collsions,
                    "path length": path_length,
                    "computation time full [in ms]": comp_time_full / 1000,
                    "computation time minimal [in ms]": comp_time_min / 1000,
                    "jerk": np.nanmean(jerk),
                    "acceleration": np.nanmean(accelerations),
                    "velocities": np.nanmean(velocities),
                    "roughness": np.nanmean(roughness),
                    "curvature": np.nanmean(curvature),
                    "normalized": np.nanmean(normalized_curvature),
                    "max curvature": np.nanmean(max_curvature),
                    "angle over length": np.nanmean(aol),
                }
            
            data = data.append(stats, ignore_index=True)
    
    return data

def main():
    args = get_args()
    data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.folder)
    files = glob.glob(f"{data_folder}/*.csv")
    data = import_data(files)

    data = data[data["path length"] > 0]
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "joined_data_stats.csv")
    data.to_csv(save_path)


if __name__ == "__main__":
    main()