from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    return parser.parse_args()



def window_smoothing(data, window_size = 3):
    return data.rolling(window_size, center=True, min_periods=2).mean()

def plot_data_stage(x, y, stage, smoothing=False, title="Plot", xlabel="x", ylabel="y", y_min="min", y_max="max", threshold_line=False, filename="test"):
    if smoothing:
        y = window_smoothing(y)
    
    if y_max == "max":
        y_max = max(y)
        y_max = y_max*1.1 if y_max > 0 else y_max * 0.9
    if y_min == "min":
        y_min = min(y) * 0.97
        y_min = y_min*1.1 if y_min < 0 else y_min * 0.9


    x_min = 0
    x_max = max(x)
    # n = stage.shape[0]
    # stage = np.append(stage, stage)
    # stage = stage.reshape((2,n)).T.reshape((2,n))
    fig = plt.figure(figsize=(28/2.54, 10/2.54))
    ax = plt.axes()
    plt.plot(x,y, color="black")
    plt.plot(x,window_smoothing(y, window_size=5), color="red", alpha=0.5)
    
    print(stage)
    print(y[np.argmax(stage):])
    y_cut_index = np.argmax(stage)
    y_cut = y[np.argmax(stage):]
    
    x_max_value = (np.argmax(y[y_cut_index:])+y_cut_index+1)*14400*32
    plt.vlines(x_max_value, y_min, y_max, colors="red", linestyles="dotted", label="best value")
    plt.ylim((y_min, y_max))
    plt.xlim((x_min,x_max))
    #print(stage[np.newaxis])
    #ax.pcolorfast((0,max(x)), ax.get_ylim(), stage.values[np.newaxis], alpha=0.3)
    stage = np.insert(np.array(stage), [0], [1])
    xv, yv = np.meshgrid(np.linspace(0,max(x), len(stage)),np.linspace(y_min, y_max, 100))
    stage = np.tile(stage, (len(yv), 1))
    print(stage)
    plt.contourf(xv, yv, stage, range(0,8), alpha=0.35)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Training Stage", rotation=270, labelpad=15)
    if threshold_line:
        plt.hlines(0.8,0,max(x),label="next stage threshold", linestyles="--", alpha=0.5)
        ax.annotate(s='next stage threshold', xy=(x_max/2, 0.8), xytext=((x_max/2)- 10, 1.05), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(["raw success rate", "smoothed success rate (rolling window avg.)"], loc="lower right")
    #plt.show()
    plt.savefig(filename, format="png")

def main():
    args = parse_args()
    data = pd.read_csv(args.csv_file)
    agent_num = int((args.csv_file.split("_")[-1]).split(".csv")[0])

    plot_data_stage(data["training step"], data["success rate"], data["stage"], title=f"Success Rate Development of Agent {agent_num}", xlabel="Training Steps", ylabel="Success Rate", y_min=0, y_max=1.1, threshold_line=True, filename=f"agent_{agent_num}_success_rate.png")
    plot_data_stage(data["training step"], data["mean reward"], data["stage"], title=f"Mean Reward Development of Agent {agent_num}", xlabel="Training Steps", ylabel="Mean Reward", y_min=-30, y_max=60, filename=f"agent_{agent_num}_mean_reward.png")


if __name__ == "__main__":
    main()
