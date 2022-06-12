import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
import matplotlib.image as mpimg
import ast

FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plots")
RESOLUTION_FACTOR = 0.11

def rotate(p, degrees=0, origin=[125,125]):
    """ rotate point """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def transform_point(p):
    """ 
        a transformation of a point to map image coordinates. 
        recorded points are in a different coordinate system.
    """
    p_scaled = np.array(p)/RESOLUTION_FACTOR
    p_flipped = np.flipud(p_scaled)
    return rotate(p_flipped, 180)

def plot_done_reasons(data:pd.DataFrame):
    first_row = data.iloc[0]
    map_type = first_row["map type"]
    num_obstacles = first_row["number obstacles"]

    fig = plt.figure()
    fig.set_size_inches(16, 9)
    
    sns.set(font_scale=2)
    ax = sns.countplot(data=data, x="done reason", hue="agent")
    ax.set(xlabel="number of episodes", ylabel="done reasons", title=f"Distribution of Done Reasons\n{map_type}, {num_obstacles} obstacles")
    plt.legend(loc="upper right")
    plt.ylim((0,150))    
    plt.savefig(os.path.join(FOLDER, f"all_agents_{map_type}_{num_obstacles}_done_reasons.png"), format="png")
    plt.close()

def plot_path_length(data:pd.DataFrame):
    first_row = data.iloc[0]
    agent_name = first_row["agent"]
    map_type = first_row["map type"]
    num_obstacles = first_row["number obstacles"]
    fig = plt.figure()
    
    sns.set(font_scale=2)
    ax = sns.catplot(data=data, x="path length", y="success", kind="box", orient="h", hue="agent", aspect=2, height=7)
    ax.set(xlabel="path length", ylabel="success", title=f"Path Length by Success\n{map_type}, {num_obstacles} obstacles")
    plt.xlim((0,130))
    plt.savefig(os.path.join(FOLDER, f"all_agents_{map_type}_{num_obstacles}_path_length.png"), format="png", bbox_inches="tight")
    plt.close()

def plot_smoothness(data:pd.DataFrame):
    first_row = data.iloc[0]
    agent_name = first_row["agent"]
    name = agent_name.replace(" ", "_")
    
    # plot jerk
    fig = plt.figure(figsize=(10,10))
    sns.set(font_scale=2)
    ax = sns.violinplot(data=data, x="map type and num obstacles", y="jerk")
    ax.set(xlabel="map type and num obstacles", ylabel="jerk", title=f"Jerk for {agent_name}")
    plt.ylim((-0.1,0.6))
    plt.savefig(os.path.join(FOLDER, f"{name}_jerk.png"), format="png", bbox_inches="tight")
    plt.close()

    # plot niormalized curvature
    fig = plt.figure(figsize=(10,10))
    sns.set(font_scale=2)
    ax = sns.violinplot(data=data, x="map type and num obstacles", y="normalized")
    ax.set(xlabel="map type and num obstacles", ylabel="normalized curvature", title=f"Curvature for {agent_name}")
    plt.ylim((-0.1,1.1))
    plt.savefig(os.path.join(FOLDER, f"{name}_curvature.png"), format="png", bbox_inches="tight")
    plt.close()

    # plot aol plot
    fig = plt.figure(figsize=(10,10))
    sns.set(font_scale=2)
    ax = sns.violinplot(data=data, x="map type and num obstacles", y="angle over length")
    ax.set(xlabel="map type and num obstacles", ylabel="angle over length", title=f"Angle Over Length for {agent_name}")
    plt.ylim((-5,40))
    plt.savefig(os.path.join(FOLDER, f"{name}_angle_over_length.png"), format="png", bbox_inches="tight")
    plt.close()

def plot_comparisons(data:pd.DataFrame):
    first_row = data.iloc[0]
    map_type = first_row["map_type"]
    num_obstacles = first_row["num_obs"]

    # plot barplot of success rates
    fig = plt.figure()
    sns.set(font_scale=1.5)
    ax = sns.catplot(data=data, kind="bar", x="agent", y="success_rate")
    ax.set(xlabel="planner", ylabel="success rate", title=f"Success Rate\n{map_type}, {num_obstacles} obstacles")
    plt.xticks(rotation=90)
    plt.ylim((0,1.1))
    plt.savefig(os.path.join(FOLDER, f"{map_type}_{num_obstacles}_success_rate.png"), format="png", bbox_inches="tight")
    plt.close()

    # plot barplot of speed
    fig = plt.figure()
    data["speed"] = data["mean_success_distance"] / data["mean_success_ep_time"]
    sns.set(font_scale=1.5)
    ax = sns.catplot(data=data, kind="bar", x="agent", y="speed")
    ax.set(xlabel="planner", ylabel="mean speed", title=f"Mean Speed\n{map_type}, {num_obstacles} obstacles")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(FOLDER, f"{map_type}_{num_obstacles}_speed.png"), format="png", bbox_inches="tight")
    plt.close()

    # plot barplot of mean path length of successful runs
    fig = plt.figure()
    sns.set(font_scale=1.5)
    ax = sns.catplot(data=data, kind="bar", x="agent", y="mean_success_distance")
    ax.set(xlabel="planner", ylabel="mean path length", title=f"Mean Success Path Length\n{map_type}, {num_obstacles} obstacles")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(FOLDER, f"{map_type}_{num_obstacles}_path_length.png"), format="png", bbox_inches="tight")
    plt.close()

    # plot mean episode time of successful runs
    fig = plt.figure()
    sns.set(font_scale=1.5)
    ax = sns.catplot(data=data, kind="bar", x="agent", y="mean_success_ep_time")
    ax.set(xlabel="planner", ylabel="mean episode time", title=f"Mean Success Episode Time\n{map_type}, {num_obstacles} obstacles")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(FOLDER, f"{map_type}_{num_obstacles}_mean_time.png"), format="png", bbox_inches="tight")
    plt.close()

    # plot barplot of mean collisions
    fig = plt.figure()
    sns.set(font_scale=1.5)
    ax = sns.catplot(data=data, kind="bar", x="agent", y="mean_collided_once")
    ax.set(xlabel="planner", ylabel="mean collisions", title=f"Mean Collisions\n{map_type}, {num_obstacles} obstacles")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(FOLDER, f"{map_type}_{num_obstacles}_mean_collisions.png"), format="png", bbox_inches="tight")
    plt.close()

def plot_qualitative_plots(folder):
    files = glob.glob(folder)

    for file in files:
        global RESOLUTION_FACTOR 
        RESOLUTION_FACTOR = 0.11
        # read csv data and clean data
        data_qual = pd.read_csv(file)
        data_qual = data_qual.iloc[1:]
        data_qual = data_qual[data_qual["episode"] < 5]

        # prepare some meta data
        file_split = file.split("/")[-1].split("_")
        agent_name = f"Agent {file_split[1]}"
        map_type = file_split[4]
        num_obs = int(file_split[-1].split(".")[0])

        # define basic figure params
        fig = plt.figure()
        fig.set_size_inches(8, 8)
        fig.set_dpi(120)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        ax.grid(False)
        plt.title(f"Paths for {map_type}, {num_obs} obstacles [{agent_name}]", fontsize=20)
        legend_elements = []
        
        # plot map
        img = np.array(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"map_{map_type}_{num_obs}_obs.png")))
        img = ndimage.rotate(img, 90)
        ax.imshow(img)

        first_row = data_qual.iloc[0]

        # plot goal and start position
        goal = transform_point([first_row["goal_pos_x"], first_row["goal_pos_y"]])
        start= transform_point([first_row["robot_pos_x"], first_row["robot_pos_y"]])
        goal = ax.scatter([goal[0]], [goal[1]], marker="*", color="brown", s=100, label="goal")
        start = ax.scatter([start[0]], [start[1]], marker="^", color="brown", s=100, label="start")

        # define colors
        colors_agent = [(250/256, 17/256, 48/256),(194/256, 12/256, 36/256),(130/256, 9/256, 25/256),(94/256, 7/256, 18/256),(61/256, 5/256, 12/256)]
        colors_obs = [(25/256, 165/256, 252/256),(19/256, 127/256, 194/256),(14/256, 90/256, 138/256),(10/256, 65/256, 99/256),(6/256, 38/256, 59/256)]

        # prepare obstacle start position
        obs_x = ast.literal_eval(first_row["obs_x"])
        obs_y = ast.literal_eval(first_row["obs_y"])
        obs_points = [transform_point([x[0], x[1]]) for x in zip(obs_x, obs_y)]

        c = None
        # plot dot for starting positions of obstacles
        for i in range(num_obs):
            c = plt.Circle(obs_points[i], 1.0, color=colors_obs[-1], fill = True, alpha = 0.6, label="Obstacle Start Position")
            ax.add_patch(c)

        legend_elements.append(start)
        legend_elements.append(goal)
        #if c is not None:
            #legend_elements.append(c)

        # iterate through episodes
        coll_legend = None
        for i in range(5):
            
            # collect data for episode
            episode_data = data_qual[data_qual["episode"] == i]
            n = len(episode_data[episode_data["reached_goal"] == True])

            #prepare robot path data
            robot_path_x = episode_data["robot_pos_x"].values.tolist()
            robot_path_y = episode_data["robot_pos_y"].values.tolist()
            robot_path = [transform_point([x[0], x[1]]) for x in zip(robot_path_x, robot_path_y)]
            robot_path_x = [x[0] for x in robot_path]
            robot_path_y = [x[1] for x in robot_path]
            
            # plot robot path
            line = ax.plot(robot_path_x, robot_path_y, "-b", color=colors_agent[i], label=f"robot run {i+1}")
            legend_elements.append(line[0])

            # get all obstacle paths
            obs_x = episode_data["obs_x"].values.tolist()
            obs_x = np.array([ast.literal_eval(x) for x in obs_x])
            obs_y = episode_data["obs_y"].values.tolist()
            obs_y = np.array([ast.literal_eval(x) for x in obs_y])

            # only plot first 2 episodes
            if i < 2:

                #iterate through obstacles
                for j in range(num_obs):    

                    # prepare obstacles path
                    single_obs_x = obs_x[:,j]
                    single_obs_y = obs_y[:,j]
                    obs_path = [transform_point([x[0], x[1]]) for x in zip(single_obs_x, single_obs_y)]
                    single_obs_x = [x[0] for x in obs_path]
                    single_obs_y = [x[1] for x in obs_path]
                    
                    # plot obstacles path but legend entry only for first obstacle
                    if j == 0: 
                        line = ax.plot(single_obs_x, single_obs_y, "-b", color=colors_obs[i], label=f"obstacles run {i+1}", alpha=0.6)
                        legend_elements.append(line[0])
                    else:
                        ax.plot(single_obs_x, single_obs_y, "-b", color=colors_obs[i])

            # iterate through laser scan to identify time point of collision
            for _, row in episode_data.iterrows():
                scan = float(row["laser_scan"].split("[")[-1].split("]")[0])
                # detect collision
                if scan <= 0.3:
                    #prepare collision point
                    point = transform_point([row["robot_pos_x"], row["robot_pos_y"]])
                    # plot collision
                    c = plt.Circle(point, 3, color="green", fill = True, alpha = 0.6, label="collision")
                    ax.add_patch(c)
                    coll_legend = c

        # collision legend entry shall be last one
        if coll_legend is not None:
            legend_elements.append(coll_legend)

        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False) 
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.6,1), fontsize=16)
        plt.savefig(os.path.join(FOLDER, f"{agent_name}_{map_type}_{num_obs}_qualiative.png"), format="png", bbox_inches="tight")
        plt.close()


def main():
    
    file_qunatitative = os.path.join(os.path.dirname(os.path.realpath(__file__)), "joined_data_stats.csv")
    file_quantitative_comparison = os.path.join(os.path.dirname(os.path.realpath(__file__)), "full_stats_quantitative.csv")
    folder_qualitative = os.path.join(os.path.dirname(os.path.realpath(__file__)), "marvin_data_qualitative", "*.csv")

    # prepare output folder
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    # prepare data 
    data_quantitative = pd.read_csv(file_qunatitative, index_col=0)
    data_quantitative["done reason"] = data_quantitative["done reason"].astype("category")
    data_quantitative["agent"] = data_quantitative["agent"].astype("category")
    data_quantitative["map type and num obstacles"] = [f"{x[0]}, {x[1]} obstacles" for x in  zip(data_quantitative["map type"].values.tolist(), data_quantitative["number obstacles"].values.tolist())]
    data_quantitative.loc[data_quantitative["map type and num obstacles"] == "indoor, 5 obstacles", "map type and num obstacles"] = "in, 5"
    data_quantitative.loc[data_quantitative["map type and num obstacles"] == "indoor, 15 obstacles", "map type and num obstacles"] = "in, 15"
    data_quantitative.loc[data_quantitative["map type and num obstacles"] == "outdoor, 5 obstacles", "map type and num obstacles"] = "out, 5"
    data_quantitative.loc[data_quantitative["map type and num obstacles"] == "outdoor, 15 obstacles", "map type and num obstacles"] = "out, 15"
    data_quantitative["map type and num obstacles"] = data_quantitative["map type and num obstacles"].astype("category")
    
    # split data by map type and number obstacles
    data_in_5 = data_quantitative[(data_quantitative["map type"] == "indoor") & (data_quantitative["number obstacles"] == 5)]
    data_in_15 = data_quantitative[(data_quantitative["map type"] == "indoor") & (data_quantitative["number obstacles"] == 15)]
    data_out_5 = data_quantitative[(data_quantitative["map type"] == "outdoor") & (data_quantitative["number obstacles"] == 5)]
    data_out_15 = data_quantitative[(data_quantitative["map type"] == "outdoor") & (data_quantitative["number obstacles"] == 15)]

    # plot quantitave data 
    print("plot done reasons")
    plot_done_reasons(data_in_5)
    plot_done_reasons(data_in_15)
    plot_done_reasons(data_out_5)
    plot_done_reasons(data_out_15)

    print("plot path lengths")
    plot_path_length(data_in_5)
    plot_path_length(data_in_15)
    plot_path_length(data_out_5)
    plot_path_length(data_out_15)

    print("plot smoothness")
    for agent_i in range(1,7):
        agent_name = f"Agent {agent_i}"
        data_agent = data_quantitative[data_quantitative["agent"] == agent_name]
        plot_smoothness(data_agent)

    # plot comparisons plots with aio
    data2 = pd.read_csv(file_quantitative_comparison, index_col=0)
    data2_in_5 = data2[(data2["map_type"] == "indoor") & (data2["num_obs"] == 5)]
    data2_in_15 = data2[(data2["map_type"] == "indoor") & (data2["num_obs"] == 15)]
    data2_out_5 = data2[(data2["map_type"] == "outdoor") & (data2["num_obs"] == 5)]
    data2_out_15 = data2[(data2["map_type"] == "outdoor") & (data2["num_obs"] == 15)]
    
    print("plot comparisons")
    plot_comparisons(data2_in_5)
    plot_comparisons(data2_in_15)
    plot_comparisons(data2_out_5)
    plot_comparisons(data2_out_15)

    # plot qualitatice plots
    print("plot qualitative plots")
    plot_qualitative_plots(folder_qualitative)
        
if __name__ == "__main__":
    main()