import numpy as np
import matplotlib.pyplot as plt

def plot_curve(data_list, filepath="./my_plot.png", x_label="X", y_label="Y",
               x_range=(0, 1), y_range=(0,1), color="-r", kernel_size=50,
               alpha=0.4, grid=True):
        """Plot a graph using matplotlib

        """
        if(len(data_list) <=1): return
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
        ax.grid(grid)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # The original data is showed in background
        ax.plot(data_list, color, alpha=alpha)
        # Smooth the graph using a convolution
        kernel = np.ones(int(kernel_size))/float(kernel_size)
        tot_data = len(data_list)
        lower_boundary = int(kernel_size/2.0)
        upper_boundary = int(tot_data-(kernel_size/2.0))
        data_convolved_array = np.convolve(data_list, kernel, 'same')[lower_boundary:upper_boundary]
        #print("arange: " + str(np.arange(tot_data)[lower_boundary:upper_boundary]))
        #print("Convolved: " + str(np.arange(tot_data).shape))
        ax.plot(np.arange(tot_data)[lower_boundary:upper_boundary],
                data_convolved_array, color, alpha=1.0)  # Convolved plot
        fig.savefig(filepath)
        fig.clear()
        plt.close(fig)
        # print(plt.get_fignums())  # print the number of figures opened in background
