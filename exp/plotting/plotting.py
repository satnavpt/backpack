"""Collection of plotting commands for plotting mean/average of runs."""

import pandas
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save


class OptimizationPlot():
    """Collection of plotting commands for optimization plots."""
    @staticmethod
    def plot_metric(csv_file,
                    plot_std=True,
                    std_alpha=0.5,
                    scale_steps=1,
                    label=None): 
        """Add plot of a metric.
        
        Parameters:
        -----------
        csv_file : (str)
            Path to the .csv file, require columns 'step', 'mean'and 'std'
        plot_std : (bool)
            Add shaded region one standard deviation around the mean
        scale_step : (float)
            Scale the steps (x-axis) by a ratio (e.g. training set size)
        alpha : (float) between 0 and 1
            Transparency of the standard deviation shade plot
        label : (str)
            Label of the plot, no label if left None
        """
        step, mean, std = OptimizationPlot.read_csv(csv_file,
                                                    scale_steps=scale_steps)
        OptimizationPlot.plot_mean(step, mean, label=label)
        if plot_std:
            OptimizationPlot.plot_std(step, mean, std, alpha=std_alpha)

    @staticmethod
    def plot_mean(steps, mean, label=None):
        """Plot the mean value."""
        plt.plot(steps, mean, label=label)

    @staticmethod
    def plot_std(steps, mean, std, alpha=0.5):
        """Plot sigma-interval around the mean.""" 
        plt.fill_between(steps, mean - std, mean + std, alpha=alpha)

    @staticmethod
    def read_csv(csv_file, scale_steps=1):
        """Read CSV summmary file, return step, mean, std."""
        data = pandas.read_csv(csv_file)
        step = data['step'] / scale_steps
        mean = data['mean']
        std = data['std']
        return step, mean, std

    @staticmethod
    def save_as_tikz(out_file, pdf_preview=True):
        """Save TikZ figure using matplotlib2tikz. Optional PDF out."""
        tex_file, pdf_file = ['{}.{}'.format(out_file, extension)
                              for extension in ['tex', 'pdf']]
        tikz_save(tex_file, 
                  override_externals = True,
                  # define these two macros in your .tex document
                  figureheight = r'\figheight',
                  figurewidth = r'\figwidth',
                  tex_relative_path_to_data = '../../fig/',
                  extra_axis_parameters = {'mystyle'})
        if pdf_preview:
            plt.savefig(pdf_file, bbox_inches='tight')

    @staticmethod
    def create_standard_plot(xlabel,
                             ylabel,
                             csv_files,
                             labels,
                             scale_steps=1,
                             plot_std=True,
                             std_alpha=0.5
                             ):
        """Standard plot of the same metric for different optimizers."""
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for label, csv in zip(labels, csv_files):
            OptimizationPlot.plot_metric(csv,
                                         plot_std=plot_std,
                                         std_alpha=std_alpha,
                                         scale_steps=scale_steps,
                                         label=label)
