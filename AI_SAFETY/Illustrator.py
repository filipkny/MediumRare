import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from matplotlib.cm import get_cmap
mpl.rc('image', cmap='viridis')

class Illustrator():
    def __init__(self):
        pass

    def print_best_policies(self, Qmat, size, supervision=False):
        best_policies = []
        sum_best_policies = []
        for y in range(size):
            sum_best_policies.append([])
            best_policies.append([])
            for x in range(size):
                key = str([[y, x], supervision])
                choices = Qmat[key]
                max_dir = max(choices, key=choices.get)

                best_policies[y].append(max_dir)
                sum_best_policies[y].append(round(max(choices.values()), 1))

        best_policies = np.matrix(best_policies)
        sum_best_policies = np.matrix(sum_best_policies)

        return best_policies, sum_best_policies

    def print_final_policies(self,snake_pit, treasure, world, walls, Qmat, supervision=False):
        best_policies, sum_best_policies = self.print_best_policies(Qmat, world.size, supervision=supervision)
        for wall in walls:
            best_policies[wall[0], wall[1]] = 'WALL'
            sum_best_policies[wall[0], wall[1]] = '-1'

        best_policies[snake_pit[0], snake_pit[1]] = 'SNAKE'
        # sum_best_policies[snake_pit[0],snake_pit[1]] = -0.000

        best_policies[treasure[0], treasure[1]] = 'TREASURE'
        sum_best_policies[treasure[0], treasure[1]] = 20

        return Qmat, best_policies, sum_best_policies

    def show_heatmap(self, treasure, walls, best_q_mat, annot=True):
        # Hardcode our walls, treasure and snake_pit values
        print(best_q_mat)
        for wall in walls:
            best_q_mat[wall[0], wall[1]] = None

        best_q_mat[treasure[0], treasure[1]] = None
        # best_q_mat[snake_pit[0], snake_pit[1]] = None
        cmap = get_cmap()
        ax = sns.heatmap(best_q_mat, annot=annot, linewidths=0.5, cmap=cmap)
        plt.show()

    def create_arrowmap(self,policy):
        from matplotlib import colors
        policy_raw = copy.copy(policy)

        policy[policy == 'WALL'] = 2
        policy[policy == 'TREAS'] = 3
        policy[policy == 'SNAKE'] = 4
        policy[policy == 'east'] = 1
        policy[policy == 'west'] = 1
        policy[policy == 'south'] = 1
        policy[policy == 'north'] = 1
        policy = policy.astype(float)

        # create discrete colormap
        cmap = colors.ListedColormap(['blue', 'black', 'green', 'red'])
        bounds = [1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #
        arrow_dir = {
            "south": ((0, 0.5), (0, -0.5)),
            "north": ((0, -0.5), (0, 0.5)),
            "east": ((0.5, 0), (-0.5, 0)),
            "west": ((-0.5, 0), (0.5, 0))
        }
        fig, ax = plt.subplots()
        ax.imshow(policy, cmap=cmap, norm=norm)
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                if policy_raw[i, j] != "WALL" and \
                        policy_raw[i, j] != "TREAS" and \
                        policy_raw[i, j] != "SNAKE":
                    dir = arrow_dir[policy_raw[i, j]]
                    xy = (dir[0][0] + j, dir[0][1] + i)
                    xytest = (dir[1][0] + j, dir[1][1] + i)
                    ax.annotate('', xy=xy, xytext=xytest,
                                arrowprops={'arrowstyle': '->', 'lw': 3, 'color': 'black'},
                                va='center')
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, policy.shape[0], 1))
        ax.set_yticks(np.arange(-.5, policy.shape[1], 1))

        plt.show()

# iters = 200000
# (rewards_sarsa,rewards_qlearn),(best_qs_qlearn,best_qs_sarsa),(best_ps_qlearn,best_ps_sarsa) = gather_data(iters,[1,100,1000,iters-1])
#
# # Plot reward convergence
# window_width = 200
# cumsum_vec_sarsa = np.cumsum(np.insert(rewards_sarsa, 0, 0))
# ma_vec_sarsa = (cumsum_vec_sarsa[window_width:] - cumsum_vec_sarsa[:-window_width]) / window_width
#
# cumsum_vec_qlearn = np.cumsum(np.insert(rewards_qlearn, 0, 0))
# ma_vec_qlearn = (cumsum_vec_qlearn[window_width:] - cumsum_vec_qlearn[:-window_width]) / window_width
#
# plt.plot(range(len(ma_vec_sarsa)),ma_vec_sarsa,label = "SARSA",linewidth=1)
# plt.plot(range(len(ma_vec_qlearn)),ma_vec_qlearn,label = "Q-learn",linewidth=1)
# plt.legend()
# plt.show()
#
# for i in range(len(best_qs_sarsa)):
#     print("{}/{}".format(i,len(best_qs_sarsa)))
#     show_heatmap(best_qs_sarsa[i], annot=True)
# #
# # for i in range(len(best_qs_qlearn)):
# #     show_heatmap(best_qs_qlearn[i])
#
# for i in range(len(best_ps_sarsa)):
#     create_arrowmap(best_ps_sarsa[i])
# #
# # for i in range(len(best_ps_qlearn)):
# #     create_arrowmap(best_ps_qlearn[i])
#
#
