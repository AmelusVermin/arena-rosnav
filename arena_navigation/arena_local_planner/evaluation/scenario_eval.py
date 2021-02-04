# for data
import rosbag
import bagpy
from bagpy import bagreader
import pandas as pd
import json
import rospkg
# for plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
from matplotlib.patches import Polygon
import matplotlib.cm as cm
# calc
import numpy as np
import math
import seaborn as sb
import rospy 
from visualization_msgs.msg import Marker, MarkerArray
import pathlib
import os

# 
class newBag():
    def __init__(self, planner,file_name, bag_name):
        # bag topics
        odom_topic="/sensorsim/police/odom"
        collision_topic="/sensorsim/police/collision"
        # 
        self.col_zones = {}
        self.nc_total = 0
        self.nc_curr = 0
        # eval bags

        self.bag = bagreader(bag_name)
        eps = self.split_runs(odom_topic, collision_topic)
        self.evalPath(planner,file_name,eps)
        # return


    def make_txt(self,file,msg,ron="a"):
        # f = open(file, ron)
        # f.write(msg)
        # f.close()
        return

    def make_heat_map(self,xy):
        # fig, ax = plt.subplots(figsize=(6, 7))
        # data = np.zeros((33, 25))

        # for xya in xy:
        #     for pos in xya:
        #         y = -int(round(pos[0], 0))
        #         x = int(round(pos[1], 0))
        #         data[x,y] += 1
        # #         # print(data[x,y])
        # # heat_map = sb.heatmap(data,  cmap="YlGnBu")
        # # heat_map.invert_yaxis()
        # # plt.show()

        # # delta = 0.025
        # # x = y = np.arange(-3.0, 3.0, delta)
        # # X, Y = np.meshgrid(x, y)
        # # Z1 = np.exp(-X**2 - Y**2)
        # # Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
        # # Z = (Z1 - Z2) * 2

        # im = ax.imshow(data, interpolation='bilinear', cmap=cm.RdYlGn,
        #        origin='lower', extent=[-3, 3, -3, 3]
        #        ,vmax=abs(data).max(), vmin=-abs(data).max())
        # plt.show()
        return 

    def split_runs(self,odom_topic,collision_topic):
        # get odometry
        
        odom_csv = self.bag.message_by_topic(odom_topic)
        df_odom = pd.read_csv(odom_csv, error_bad_lines=False)

        # get collision time
        try:
            # check if collision was published
            collision_csv = self.bag.message_by_topic(collision_topic)
            df_collision = pd.read_csv(collision_csv, error_bad_lines=False)
        except Exception as e:
            # otherwise run had zero collisions
            df_collision = []
        t_col = []

        for i in range(len(df_collision)): 
            t_col.append(df_collision.loc[i, "Time"])
        self.nc_total = len(t_col)
        # get reset time
        reset_csv = self.bag.message_by_topic("/scenario_reset")
        df_reset = pd.read_csv(reset_csv, error_bad_lines=False)
        t_reset = []
        for i in range(len(df_reset)): 
            t_reset.append(df_reset.loc[i, "Time"])
        # print(t_reset)

        pose_x = []
        pose_y = []
        t = []
        bags = {}
        # run idx
        n = 0
        # collsion pos
        col_xy = []
        nc = 0


        for i in range(len(df_odom)): 
            current_time = df_odom.loc[i, "Time"]
            x = df_odom.loc[i, "pose.pose.position.x"]
            y = df_odom.loc[i, "pose.pose.position.y"]
            reset = t_reset[n]
            # check if respawned
            # if current_time > reset-6 and n < len(t_reset)-1 and x<0:
            global start
            start_x = start[0] + 0.5
            if current_time > reset-6 and n < len(t_reset)-1 and x < start_x:
                n += 1
                # store the run
                bags["run_"+str(n)] = [pose_x,pose_y,t,col_xy]

                # reset 
                pose_x = []
                pose_y = []
                t = []
                col_xy = []
   
            if  len(pose_x) > 0:
                pose_x.append(x)
                pose_y.append(y)
            elif x < start_x:
                # print("run_"+str(n))
                # print("x:",x,"y",y)
                pose_x.append(x)
                pose_y.append(y)

            t.append(current_time)
            # get trajectory

            # check for col
            if len(t_col) > nc:
                if current_time >= t_col[nc]:
                    col_xy.append([x,y])
                    nc += 1

        if "run_1" in bags:    
            bags.pop("run_1")
        return bags
    
    def average(self,lst): 
        if len(lst)>0:
            return sum(lst) / len(lst) 
        else:
             return 0

    def plot_collisions(self, xya, clr, multiple = True):
        global ax

        z_id = 0
        if multiple:
            for run_a in xya:
                for col_xy in run_a:
                    circle = plt.Circle((-col_xy[1], col_xy[0]), 0.3, color=clr, fill = True, alpha = 0.3)
                    ax.add_patch(circle)
                    # if z_id in self.col_zones:
                    #     self.col_zones[z_id] = []
                    # else:
                    #     self.col_zones[z_id] = [col_xy[0], col_xy[1], 0]
        else:
            if self.nc_curr > 0:
                circle.remove()

            col_xy = xya
            self.nc_curr += 0.05
            circle = plt.Circle((-col_xy[1], col_xy[0]), self.nc_curr, color=clr, fill = False, alpha = 0.3)
            ax.add_patch(circle)

    def evalPath(self, planner, file_name, bags):
        col_xy = []
        global ax, lgnd, axlim

        durations = [] 
        trajs = []
        vels  = []

        self.make_txt(file_name, "\n"+"Evaluation of "+planner+":")
        axlim = {}
        axlim["x_min"] = 100
        axlim["x_max"] = -100
        axlim["y_min"] = 100
        axlim["y_max"] = -100

        for run in bags:
            if run != "nrun_30":
                pose_x = bags[run][0]
                pose_y = bags[run][1]

                x = np.array(pose_x)
                y = -np.array(pose_y)
                # # x
                # if min(x) < axlim["x_min"]:
                #     axlim["x_min"] = x
                # if max(x) > axlim["x_max"]:
                #     axlim["x_max"] = x
                # # y
                # if min(y) < axlim["y_min"]:
                #     axlim["y_min"] = y
                # if max(y) > axlim["y_max"]:
                #     axlim["y_max"] = y
                
                
                t = bags[run][2]

                dist_array = (x[:-1]-x[1:])**2 + (y[:-1]-y[1:])**2
                path_length = np.sum(np.sqrt(dist_array)) 
                # for av
                trajs.append(path_length)
                if path_length > 0:
                    ax.plot(y, x, lgnd[planner], alpha=0.2)


                duration = t[len(t)-1] - t[0]
                # for av
                durations.append(duration)
                av_vel = path_length/duration
                # for av
                vels.append(av_vel)

                n_col = len(bags[run][3])

                duration    = round(duration,3)
                path_length = round(path_length,3)
                av_vel      = round(av_vel,3)

                cr = run+": "+str([duration, path_length, av_vel, n_col])
                print(cr)
                self.make_txt(file_name, "\n"+cr)

                col_xy.append(bags[run][3])

                # if len(bags[run][3]) > 0:
                #     self.plot_collisions(bags[run][3][0], lgnd[planner], False)



        msg_planner = "\n----------------------   "+planner+" summary: ----------------------"
        msg_at = "\naverage time:        "+str(round(self.average(durations),3))+" s"
        msg_ap = "\naverage path length: "+str(round(self.average(trajs),3))+" m"
        msg_av = "\naverage velocity:    "+str(round(self.average(vels),3))+"  m/s"
        msg_col = "\ntotal number of collisions: "+str(self.nc_total)+"\n"

        print("----------------------   "+planner+"   ----------------------")
        print("average time:        ", round(self.average(durations),3), "s")
        print("average path length: ", round(self.average(trajs),3), "m")
        print("average velocity:    ", round(self.average(vels),3), " m/s")
        print("total collisions:    ",  str(self.nc_total))

        self.make_txt(file_name,msg_planner)
        self.make_txt(file_name,msg_at)
        self.make_txt(file_name,msg_ap)
        self.make_txt(file_name,msg_av)
        self.make_txt(file_name,msg_col)

        self.plot_collisions(col_xy,lgnd[planner])
        # self.make_heat_map(col_xy)

def plot_arrow(start,end):
    global ax
    # ax.arrow(-start[1], start[0], -end[1], end[0], head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.arrow(-start[1], start[0], -end[1], end[0],  
        head_width = 0.2, 
        width = 0, 
        ec = "black",
        fc = "black",
        ls ="-")

def plot_dyn_obst(ob_xy):
    global ax

    circle = plt.Circle((-ob_xy[1], ob_xy[0]), 0.3, color="black", fill = False, alpha = 1)
    ax.add_patch(circle)

def read_scn_file(map, ob):
    global start, goal
    # find json path
    rospack = rospkg.RosPack()
    json_path = rospack.get_path('simulator_setup')+'/scenarios/eval/'
    print(json_path)
    for file in os.listdir(json_path):
        if file.endswith(".json") and map in file and ob in file:
            jf = file
    # read file
    with open(json_path+"/"+jf, 'r') as myfile:
        data=myfile.read()
    obj = json.loads(data)

    # json to dict
    for i in obj:
        for l in obj[i]:
            data = l
    
    # get json data
    for do in data["dynamic_obstacles"]:
        sp   = data["dynamic_obstacles"][do]["start_pos"]
        sp_x = sp[0]
        sp_y = sp[1]

        wp   = data["dynamic_obstacles"][do]["waypoints"][0]
        wp_x = wp[0]
        wp_y = wp[1]
        
        ep_x = sp_x + wp_x
        ep_y = sp_y + wp_y
        ep   = [ep_x, ep_y]

        plot_dyn_obst(sp)
        plot_dyn_obst(ep)
        plot_arrow(sp,wp)

        # print(sp)
        # print(ep)
        # print("------------------------")
        
    start = data["robot"]["start_pos"]
    goal  = data["robot"]["goal_pos"]

def eval_all(a,map,ob,vel):
    global ax, sm, lgnd, start, goal, axlim
    fig, ax = plt.subplots(figsize=(6, 7))
    
    read_scn_file(map, ob)

    mode =  map + "_" + ob + "_" + vel 
    # fig.suptitle(mode, fontsize=16)
    # plot static map
    if not "empty" in map:
        # img = plt.imread("map_small.png")
        # ax.imshow(img, extent=[-20, 6, -6, 27.3])
        plt.scatter(sm[1], sm[0],s = 0.2 , c = "grey")
        # plt.plot(sm[1], sm[0],"--")
        
    # return
    cur_path = str(pathlib.Path().absolute())
    # print(cur_path)
    for planner in a:
        for file in os.listdir(cur_path+"/bags/scenarios/"+planner):
            if file.endswith(".bag") and map in file and ob in file and vel in file:
                # print("bags/scenarios/"+planner+"/"+file)
                fn = planner + mode
                
                newBag(planner, fn, "bags/scenarios/"+planner+"/"+file)
    
    # dhow legend labels once per planner
    legend_elements = []
    for l in lgnd:
            el = Line2D([0], [0], color=lgnd[l], lw=4, label=l)
            legend_elements.append(el)

    ax.legend(handles=legend_elements, loc=0)
    
    ax.set_ylim([start[0]-1, goal[0]+1])

    # if "map" in map:
    #     ax.set_xlim([start[0]-1, goal[0]+1])
    # else:
    #     ax.set_xlim([start[0]-1, goal[0]+1])


    # print(start)
    # print(goal)
    # ax.set_ylim([axlim["y_min"], axlim["y_max"]])
    # ax.set_xlim([axlim["x_min"], axlim["x_max"]])




    plt.savefig('plots/'+mode+'.pdf')

def getMap(msg):
    global ax, sm
    points_x = []
    points_y = []
    # print(msg.markers[0])
    for p in msg.markers[0].points:
        if  2 < p.y < 25 :
            points_x.append(p.x-6)
            points_y.append(-p.y+6)
    # plt.scatter(points_y, points_x)
    sm = [points_x, points_y]
    # plt.show()

    # map_obs ={}
    # obc = 0
    # for i in range(len(points_x)):
    #     # x = points_x[i]
    #     # y = points_y[i]
    #     # if obc not in map_obs[obc]:
    #     #     map_obs[obc] = []

    #     # map_obs[obc] = map_obs[obc].append([points_x, points_y])
    #     if i < len(points_x)-1:
    #         dx = points_x[i+1] -  points_x[i]
    #         dy = points_y[i+1] -  points_y[i]
    #         dp = math.sqrt(dx**2+dy**2)

    #         if dp > 1:
    #             print(i)
    
def run():
    global ax, sm, lgnd

    lgnd = {}
    lgnd["arena"] = "tab:purple"
    lgnd["cadrl"] = "tab:red"
    lgnd["dwa"] = "tab:blue"
    lgnd["mpc"] = "tab:green"
    lgnd["teb"] = "tab:orange"
    # static map
    rospy.init_node("eval", anonymous=False)
    rospy.Subscriber('/flatland_server/debug/layer/static',MarkerArray, getMap)
    

    # start_x = 0.5
    # map
    #  5 01
    eval_all(["arena","cadrl","dwa","mpc","teb"],"map1","5","vel_03")
    # # start_x = 0
    # #  10 01
    eval_all(["arena","cadrl","dwa","mpc","teb"],"map1","10","vel_03")
    # 20 01
    eval_all(["arena","cadrl","dwa","mpc","teb"],"map1","20","vel_03")


    # empty map
    #  5 01
    eval_all(["arena","cadrl","dwa","mpc","teb"],"empty","5","vel_03")    
    #  10 01
    eval_all(["arena","cadrl","dwa","mpc","teb"],"empty","10","vel_03")    
    #  20 01
    eval_all(["arena","cadrl","dwa","mpc","teb"],"empty","20","vel_03")

    
    
    plt.show()
    rospy.spin()


if __name__=="__main__":
    run()
