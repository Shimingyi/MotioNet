import os
import numpy as np
import importlib
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    w, h, d = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tostring())


def visual_result_grid(img_path, pose_2d, pose_3d_pre, pose_3d_gt, save_path=None):
    fig = plt.figure()

    gs1 = gridspec.GridSpec(2, 2)
    # gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    # plt.axis('off')
    # Show real image
    ax1 = plt.subplot(gs1[0, 0])
    img = Image.open(img_path)
    ax1.imshow(img)

    # Show 2d pose
    ax2 = plt.subplot(gs1[0, 1])
    show2Dpose(pose_2d, ax2)
    ax2.set_title('2d input')
    ax2.invert_yaxis()

    # Plot 3d predict
    ax3 = plt.subplot(gs1[1, 0], projection='3d')
    ax3.set_title('3d predict')
    show3Dpose(pose_3d_pre, ax3)
    # ax3.view_init(0, -90)

    # Plot 3d gt
    ax4 = plt.subplot(gs1[1, 1], projection='3d')
    ax4.set_title('3d gt')
    show3Dpose(pose_3d_gt, ax4)

    if save_path is None:
        fig_img = fig2img(fig)
        plt.close()
        return fig_img
    else:
        fig.savefig(save_path)
        plt.close()


def visual_result_row(img_path, pose_2d, pose_3d_pre, pose_3d_gt, save_path=None):
    fig = plt.figure()

    gs1 = gridspec.GridSpec(1, 4)
    # gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    # plt.axis('off')
    # Show real image
    ax1 = plt.subplot(gs1[0])
    img = Image.open(img_path)
    ax1.imshow(img)

    # Show 2d pose
    ax2 = plt.subplot(gs1[1])
    show2Dpose(pose_2d, ax2)
    ax2.set_title('2d input')
    # ax2.invert_yaxis()

    # Plot 3d predict
    ax3 = plt.subplot(gs1[2], projection='3d')
    ax3.set_title('3d predict')
    show3Dpose(pose_3d_pre, ax3, radius=1)

    # Plot 3d gt
    ax4 = plt.subplot(gs1[3], projection='3d')
    ax4.set_title('3d gt')
    show3Dpose(pose_3d_gt, ax4, radius=1)

    if save_path is None:
        fig_img = fig2img(fig)
        plt.close()
        return fig_img
    else:
        fig.savefig(save_path)
        plt.close()


def visual_sequence_result_without_image(pose_2d, pose_3d_predict, pose_3d_gt, save_path=None):
    fig = plt.figure(figsize=(60, 12))

    gs1 = gridspec.GridSpec(3, 20)
    # gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    # plt.axis('off')

    for i in range(20):

        ax2 = plt.subplot(gs1[0, i])
        show2Dpose(pose_2d[i, :], ax2, radius=500)
        # ax2.invert_yaxis()

        ax2 = plt.subplot(gs1[1, i], projection='3d')
        show3Dpose(pose_3d_predict[i, :], ax2, radius=np.max(pose_3d_gt))

        ax3 = plt.subplot(gs1[2, i], projection='3d')
        show3Dpose(pose_3d_gt[i, :], ax3, radius=np.max(pose_3d_gt))

    if save_path is None:
        fig_img = fig2img(fig)
        plt.close()
        return fig_img
    else:
        fig.savefig(save_path)


def visual_sequence_result(file_path, pose_2d, pose_3d_predict, pose_3d_gt, save_path=None):
    fig = plt.figure(figsize=(60, 12))

    gs1 = gridspec.GridSpec(4, 20)
    # gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    # plt.axis('off')

    for i in range(20):
        # Show real image
        ax1 = plt.subplot(gs1[0, i])
        img = Image.open(file_path[i])
        ax1.imshow(img)

        ax2 = plt.subplot(gs1[1, i])
        show2Dpose(pose_2d[i, :], ax2)
        ax2.invert_yaxis()

        ax2 = plt.subplot(gs1[2, i], projection='3d')
        show3Dpose(pose_3d_predict[i, :], ax2)

        ax3 = plt.subplot(gs1[3, i], projection='3d')
        show3Dpose(pose_3d_gt[i, :], ax3)

    if save_path is None:
        fig_img = fig2img(fig)
        plt.close()
        return fig_img
    else:
        fig.savefig(save_path)


def show3Dpose(channels, ax, radius=600, lcolor="#3498db", rcolor="#e74c3c", add_labels=True):  # blue, orange
    """
    Visualize a 3d skeleton

    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
    Returns
      Nothing. Draws on ax.
    """

    if channels.size == 48:
        vals = np.reshape(channels, (-1, 3))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 7, 10, 11, 7, 13, 14])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif channels.size == 51:
        vals = np.reshape(channels, (-1, 3))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif channels.size == 45:
        vals = np.reshape(channels, (-1, 3))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 7, 9, 10, 7, 12, 13])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    vals[:, [1, 2]] = vals[:, [2, 1]]
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([RADIUS + yroot, -RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
    # Get rid of the ticks and tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    # white = (1.0, 1.0, 1.0, 0.0)
    # ax.w_xaxis.set_pane_color(white)
    # ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    # ax.w_xaxis.line.set_color(white)
    # ax.w_yaxis.line.set_color(white)
    # ax.w_zaxis.line.set_color(white)


def show3Dpose_fixed(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True, label_min=None, label_max=None):  # blue, orange
    if channels.size == 48:
        vals = np.reshape(channels, (-1, 3))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 7, 10, 11, 7, 13, 14])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif channels.size == 51:
        vals = np.reshape(channels, (-1, 3))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    else:
        vals = np.reshape(channels, (-1, 3))
        I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1  # start points
        J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    vals[:, [1, 2]] = vals[:, [2, 1]]

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, z, y, lw=2, c=lcolor if LR[i] else rcolor)

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")

    if label_min is not None and label_max is not None:
        ax.set_xlim3d([label_min, label_max])
        ax.get_xaxis().set_ticklabels(list(range(int(label_min), int(label_max), int((label_max-label_min)/3))))
        ax.set_ylim3d([label_min, label_max])
        ax.get_yaxis().set_ticklabels(list(range(int(label_min), int(label_max), int((label_max-label_min)/3))))
        ax.set_zlim3d([label_min, label_max])
        ax.set_zticklabels(list(range(int(label_min), int(label_min), int((label_max-label_min)/3))))

    ax.set_aspect('equal')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)

    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def show2Dpose(channels, ax, radius=600, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    if channels.size == 32:
        vals = np.reshape(channels, (-1, 2))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 7, 10, 11, 7, 13, 14])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif channels.size == 34:
        vals = np.reshape(channels, (-1, 2))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif channels.size == 30:
        vals = np.reshape(channels, (-1, 2))
        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 7, 9, 10, 7, 12, 13])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        if np.mean(x) != 0 and np.mean(y) != 0:
            ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xticks([0, 200, 400, 600, 800, 1000])
    # ax.set_yticks([0, 200, 400, 600, 800, 1000])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    RADIUS = radius  # space around the subject
    xroot, yroot = vals[0, 0], vals[0, 1]
    ax.set_xlim([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim([-RADIUS + yroot, RADIUS + yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    ax.set_aspect('equal')


class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = """TensorboardX visualization is configured to use, but currently not installed on this machine. Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."""
                warnings.warn(message, UserWarning)
                logger.warn()
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def set_scalars(self, scalars):
        for key, value in scalars.items():
            self.add_scalar(key, value)

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr
