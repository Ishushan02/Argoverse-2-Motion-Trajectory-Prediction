import matplotlib.pyplot as plt
import matplotlib.animation as animation
from read_data import getData
import pandas as pd
import numpy as np


def make_gif(data_matrix, name='example'):
   cmap = plt.get_cmap('viridis', 50)

   fig, ax = plt.subplots(figsize=(10, 10))
   # Function to update plot for each frame
   def update(frame):
       ax.clear()

       # Get data for current timestep
       for i in range(1, data_matrix.shape[0]):
           x = data_matrix[i, frame, 0]
           y = data_matrix[i, frame, 1]
           if x != 0 and y != 0:
               xs = data_matrix[i, :frame+1, 0]  # Include current frame
               ys = data_matrix[i, :frame+1, 1]  # Include current frame
               # trim all zeros
               mask = (xs != 0) & (ys != 0)  # Only keep points where both x and y are non-zero
               xs = xs[mask]
               ys = ys[mask]

               # Only plot if we have points to plot
               if len(xs) > 0 and len(ys) > 0:
                   color = cmap(i)
                   ax.plot(xs, ys, alpha=0.9, color=color)
                   ax.scatter(x, y, s=80, color=color)
       
       ax.plot(data_matrix[0, :frame, 0], data_matrix[0, :frame, 1], color='tab:orange', label='Ego Vehicle')
    #    ax.scatter(data_matrix[0, frame, 0], data_matrix[0, frame, 1], s=80, color='tab:orange')
       # Set title with timestep
       ax.set_title(f'Timestep {frame}')
       # Set consistent axis limits
       ax.set_xlim(data_matrix[:,:,0][data_matrix[:,:,0] != 0].min() - 10, data_matrix[:,:,0][data_matrix[:,:,0] != 0].max() + 10)
       ax.set_ylim(data_matrix[:,:,1][data_matrix[:,:,1] != 0].min() - 10, data_matrix[:,:,1][data_matrix[:,:,1] != 0].max() + 10)
       ax.legend()

       return ax.collections + ax.lines

   # Create animation
   anim = animation.FuncAnimation(fig, update, frames=list(range(0, data_matrix.shape[1], 3)),
                               interval=100, blit=True)
   # Save as GIF
   anim.save(f'trajectory_visualization_{name}.gif', writer='pillow')
   plt.close()


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_vehicle_trajectory_gif(data1, data2, filename="vehicle_trajectory.gif"):
    """
    Plots the vehicle trajectory and saves it as a GIF.
    
    Parameters:
    - data1: numpy array of shape (n1, 2), first trajectory segment (x, y)
    - data2: numpy array of shape (n2, 2), second trajectory segment (x, y)
    - filename: string, name of the output GIF file
    """
    x1, y1 = data1[:, 0], data1[:, 1]
    x2, y2 = data2[:, 0], data2[:, 1]

    fig, ax = plt.subplots()
    ax.set_xlim(min(np.min(x1), np.min(x2)), max(np.max(x1), np.max(x2)) + 10)
    ax.set_ylim(min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)) + 5)
    line1, = ax.plot([], [], 'orange', label='First Segment')
    line2, = ax.plot([], [], 'blue', label='Second Segment')
    ax.legend()
    ax.set_title("Vehicle Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    total_frames = len(x1) + len(x2)

    def animate(i):
        if i < len(x1):
            line1.set_data(x1[:i+1], y1[:i+1])
            line2.set_data([], [])
        else:
            line1.set_data(x1, y1)
            line2.set_data(x2[:i - len(x1) + 1], y2[:i - len(x1) + 1])
        return line1, line2

    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=100, blit=True)
    ani.save(filename, writer='pillow')
    plt.close(fig)

# Example usage
# data1 = np.column_stack((np.linspace(0, 50, 50), 2 * np.ones(50)))
# data2 = np.column_stack((np.linspace(50, 110, 60), 2 + 0.05 * (np.linspace(0, 60, 60)**1.5)))
# plot_vehicle_trajectory_gif(data1, data2)


df = pd.read_csv("./submission/testTransFormer.csv")

train_data, test_data = getData("data")

matrix = df[['x', 'y']].values  # Shape will be (10, 2)

traj1 = matrix[:60]
print(test_data[0, 0, :, :2].shape, traj1.shape)
plot_vehicle_trajectory_gif(test_data[4, 0, :, :2], matrix[240:300], "vehicle_trajectory-2.gif")