"""Visualization utilities for distributions and targets."""
from jax import vmap, numpy as jnp
import numpy as np

from flowjax.distributions import AbstractDistribution, AbstractTransformed

from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.axes import Axes
from matplotlib.typing import ColorType, LineStyleType

import seaborn as sns

from ecg.targets.base import Target
from ecg.distributions import JointDistribution, GaussianMixture
from ecg.utils import fill_diagonal, ab_kl_div

from typing import Sequence
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray


def add_legends(axes: ArrayLike,
                dims: int,
                legend_axis: int = 0,
                labels: Sequence[str] | None = None,
                loc: str = 'upper left',
                fontsize: int = 15):
    """
    Adds a legend for histogram plots to the figure.

    - If there are unused subplots (axes beyond data dimensions), the legend
      is placed in the first unused one (and that axis is turned off).
    - Otherwise, the legend is placed on the subplot at `legend_axis`.

    Parameters
    ----------
    axes: array-like
        Flattened list or array of matplotlib Axes objects.
    D: int
        Number of data dimensions (used to identify unused subplots).
    legend_axis: int
        Index of axis to use if there are no unused subplots.
    """
    if labels is None:
        labels = ['Target', 'Prediction']

    axes = np.atleast_1d(axes).flatten()

    unused_axes = axes[dims:]

    legend_ax = unused_axes[0] if len(unused_axes) > 0 else axes[legend_axis]

    target_handle = legend_ax.plot(
        [], [], linestyle=(0, (4, 3)),
        color='tab:orange',
        linewidth=2.0,
        label=labels[0]
    )[0]

    pred_handle = legend_ax.plot([], [],
                                 color='tab:blue',
                                 linewidth=2.0,
                                 label=labels[1])[0]

    legend_ax.legend(
        handles=[target_handle, pred_handle],
        loc=loc,
        fontsize=fontsize + 2,
    )


def plot_save(fig, transparent=False, save_name=None, folder_name=None):
    if (folder_name and save_name) is not None:
        fig.savefig(f'outputs/figures/{folder_name}/{save_name}.png',
                    transparent=transparent,
                    bbox_inches='tight')
    plt.show()
    plt.close('all')


def create_1d_energy_plots(ax: Axes,
                           x_vals: Array,
                           predicted: Array,
                           ref: Array | None = None,
                           labels: Sequence[str] | None = None,
                           colors: Sequence[ColorType] | None = None,
                           x_label=r'$x_1$',
                           y_label=r'$\beta U(x_1)$',
                           y_limit=None):

    if labels is None:
        labels = ['Prediction', 'Target']

    if colors is None:
        colors = ['tab:blue', 'tab:orange']

    ax.plot(x_vals, predicted,
            label=labels[0], color=colors[0], linewidth='2')

    if ref is not None:
        ax.plot(x_vals, ref, '--',
                label=labels[1], color=colors[1], linewidth='2')

    ax.set_ylim((-2.5, 35))
    ax.set_ylim(y_limit)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def create_marginals_plots(
    axes: ArrayLike,
    samples: Array,
    ref_samples: Array | None = None,
    reference: Array | None = None,
    labels: Sequence[str] | None = None,
    y_lim: tuple[float, float] | None = None,
    axis_label_pattern: str | None = None,
    single_axis_labels: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
    legend_axis: int = 0,
    fontsize: int = 10,
    labelsize: int = 10,
    loc: str = 'upper right',
    wspace: float = 0.2,
    bins: str | int = 'auto',
):
    """
    Create marginal histograms of samples (N, dim) on given axes.
    Optionally compares to reference samples of same shape.
    """

    if labels is None:
        labels = ['Target', 'Prediction']

    if not isinstance(axes, (list, np.ndarray)):
        axes = np.atleast_1d(axes)

    if samples.ndim == 1:
        samples = jnp.expand_dims(samples, axis=-1)
    if ref_samples is not None and ref_samples.ndim == 1:
        ref_samples = jnp.expand_dims(ref_samples, axis=-1)

    dim = samples.shape[1]
    axes = axes.flatten()

    for i in range(dim):
        data_i = samples[:, i]

        hist, edge_bins = np.histogram(
            data_i,
            bins=bins,
            density=True,
            range=(
                np.min(data_i),
                np.max(data_i),
            ),
        )
        axes[i].stairs(hist, edge_bins, linewidth=2.0, color='tab:blue')

        if reference is not None:
            axes[i].plot(reference[:, 0],
                         reference[:, 1],
                         linestyle=(0, (4, 3)),
                         dashes=(4, 3),
                         color='tab:orange',
                         linewidth=2.0)

        elif ref_samples is not None:
            ref_i = ref_samples[:, i]
            hist_ref, edge_bins_ref = np.histogram(ref_i,
                                                    bins=bins,
                                                    density=True)
            axes[i].stairs(
                hist_ref,
                edge_bins_ref,
                linestyle=(0, (4, 3)),
                color='tab:orange',
                linewidth=2.0,
            )

        axes[i].tick_params(axis='x', labelsize=labelsize)
        axes[i].tick_params(axis='y', labelsize=labelsize)

        if single_axis_labels is not None:
            if i == 0:
                axes[i].set_ylabel(single_axis_labels[1], fontsize=fontsize)
            elif i == dim - 1:
                axes[i].set_xlabel(single_axis_labels[0], fontsize=fontsize)

        if axis_label_pattern is not None:
            axis_label = axis_label_pattern % (i+1)
            axes[i].set_xlabel(axis_label, fontsize=fontsize)
            axes[i].set_ylabel('$p($' + axis_label + '$)$', fontsize=fontsize)

        if titles is not None:
            axes[i].set_title(titles[i], fontsize=fontsize)

        if y_lim:
            axes[i].set_ylim(y_lim[0], y_lim[1])

    for k in range(dim, len(axes)):
        axes[k].axis('off')

    add_legends(axes,
                dim,
                labels=labels,
                loc=loc,
                fontsize=fontsize,
                legend_axis=legend_axis)

    plt.subplots_adjust(wspace=wspace)


def plot_energy_marginals(model: JointDistribution,
                          key: PRNGKeyArray,
                          target: Target,
                          beta: Float = 1.0,
                          num_samples: int = 100000,
                          resolution: int = 10000,
                          interval: float = 4.,
                          save_name: str | None = None,
                          folder_name: str | None = None,
                          ):

    x1_vals = jnp.linspace(-interval, interval, resolution)
    size_x = 6.4 * 1.5
    size_y = 2.0 * 1.5

    energy_x1 = target.energy_x1(x1_vals)
    min_position = jnp.argmin(energy_x1)
    min_energy = energy_x1[min_position]
    energy_x1_reweight = energy_x1 - min_energy

    x_vals = jnp.column_stack((x1_vals, jnp.zeros_like(x1_vals)))

    log_prob = model.log_prob(x_vals)
    pred_energy = log_prob[min_position] - log_prob

    fig, axes = plt.subplots(1,2,
                             figsize=[size_x, size_y],
                             constrained_layout=False)
    create_1d_energy_plots(axes[0], x1_vals, pred_energy,
                           energy_x1_reweight*beta)

    p_x1 = target.p_x1(x1_vals, beta)
    samples = model.sample_and_split(key, (num_samples,))[0]
    reference = jnp.stack((x1_vals, p_x1), axis=-1)
    create_marginals_plots(axes[1],
                           samples,
                           reference=reference,
                           axis_label_pattern=r'$x_{%d}$',
                           y_lim=(0, 1.5),
                           labels=[rf'Target $\beta\approx{beta:.2f}$',
                                   'Prediction'])

    plot_save(fig,
              save_name='Energy_and_Marginals_'+\
              save_name if save_name else None,
              folder_name=folder_name)


def plot_marginals(samples: Array,
                   ref_samples: Array | None = None,
                   bins: str | int = 50,
                   axis_label_pattern: str | None = None,
                   single_axis_labels: Sequence[str] | None = None,
                   titles: Sequence[str] | None = None,
                   save_name: str | None = None,
                   folder_name: str | None = None,
                   ):

    n_dims = samples.shape[1]
    n_cols = int(np.ceil(np.sqrt(n_dims)))
    n_rows = int(np.ceil(n_dims / n_cols))

    fig_size = (6.4*(1.5**(n_cols-1)), 4.8*(1.5**(n_rows-1)))

    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=fig_size,
                             constrained_layout=False)

    create_marginals_plots(
        axes,
        samples,
        ref_samples=ref_samples,
        bins=bins,
        axis_label_pattern=axis_label_pattern,
        single_axis_labels=single_axis_labels,
        titles=titles,
        fontsize=16,
        labelsize=16,
        wspace=0.3,
        loc='upper left',
    )

    plot_save(fig,
              save_name='Marginals_'+\
              save_name if save_name else None,
              folder_name=folder_name)

def plot_2d_histogram(model: AbstractDistribution,
                      key: PRNGKeyArray,
                      target: Target,
                      bins: int = 60,
                      num_samples: int = 100000,
                      save_name: str | None = None,
                      folder_name: str | None = None,
                      ):

    ref_samples = target.ref_samples()

    samples = model.sample(key, (num_samples,))

    fig, axs = plt.subplots(ncols=2, figsize=(6.4, 4.8),
                        constrained_layout=True,
                        sharey=True)
    images = []
    images.append(axs[0].hist2d(ref_samples[:,0], ref_samples[:,1], bins=bins,
            density=True, cmin=1e-10))
    axs[0].set_xlabel(r'$x_1$')
    axs[0].set_ylabel(r'$x_2$')
    axs[0].set_xlim([-4,4])
    axs[0].set_ylim([-4.5,4.5])
    axs[0].set_title('Target')

    images.append(axs[1].hist2d(samples[:,0], samples[:,1], bins=bins,
            density=True, cmin=1e-10))
    axs[1].set_xlabel(r'$x_1$')
    axs[1].set_xlim([-4,4])
    axs[1].set_ylim([-4.5,4.5])
    axs[1].set_title('Prediction')

    vmin = images[0][3].get_array().min()
    vmax = images[0][3].get_array().max()
    norm = clr.LogNorm(vmin=vmin, vmax=vmax)

    for im in images:
        im[3].set_norm(norm)
    cbar = fig.colorbar(images[0][3],ax=axs)

    plot_save(fig,
              save_name='2d_histogram_'+save_name if save_name else None,
              folder_name=folder_name)


def create_kl_contour_plot(A: Array,
                           B: Array,
                           kl_vals: Array,
                           trajectory: list[ArrayLike] | None = None,
                           save_name: str | None = None,
                           folder_name: str | None = None,
                           ):

    dim = A.shape[0]
    reverse_matrix = kl_vals.reshape((dim, dim))
    reverse_matrix = fill_diagonal(reverse_matrix, jnp.inf)

    fig = plt.figure(figsize=(6.4/1.5, 4.8/1.5))
    contour = plt.contourf(A, B, reverse_matrix, levels=50, cmap='viridis')

    if trajectory is not None:
        traj = np.asarray(trajectory)
        plt.scatter(traj[0,0], traj[0,1],
                    color='red', marker='x',
                    s=50, label='Trajectory')
        plt.plot(traj[:,0], traj[:,1],
                 color='red', marker='o',
                 linewidth=2, markersize=2)

    plt.colorbar(contour)
    plt.xlabel(r'$a_1$')
    plt.ylabel(r'$a_2$')
    plt.tight_layout()
    plot_save(fig,
              save_name='KL_contour_'+save_name if save_name else None,
              folder_name=folder_name)


def plot_kl_div(model: AbstractTransformed,
                key: PRNGKeyArray,
                target: Target,
                beta: Float = 1.0,
                trajectory: list[ArrayLike] | None = None,
                n_points: int = 100,
                num_samples: int = 1000,
                save_name: str | None = None,
                folder_name: str | None = None,
                ):

    a_vals = jnp.linspace(0, 1, n_points)
    b_vals = jnp.linspace(0, 1, n_points)
    A, B = jnp.meshgrid(a_vals, b_vals)

    zX, joint_log_prob = model.base_dist.sample_and_log_prob(key,
                                                             (num_samples,))

    reverse_kls = vmap(ab_kl_div, in_axes=(0,0,None,None,None,None))(
    A.flatten(), B.flatten(), zX, joint_log_prob.mean(), target, beta)

    create_kl_contour_plot(A, B, reverse_kls,
                           trajectory=trajectory,
                           save_name=save_name,
                           folder_name=folder_name)


def plot_matrix(A1: Array,
                A2: Array | None = None,
                titles: Sequence[str] | None = None,
                annot: bool = False,
                xticklabels: Sequence[str] | str = 'auto',
                yticklabels: Sequence[str] | str = 'auto',
                fontsize: int = 14,
                labelsize: int = 10,
                annot_size: int = 14,
                save_name: str | None = None,
                folder_name: str | None = None,
                ):

    if A2 is None:
        fig, axes = plt.subplots(1, 1, figsize=(6.4/1.5*1.1, 4.8/1.5),
                             constrained_layout=True)
        vmin = np.min(A1)
        vmax = np.max(A1)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(6.4/1.5*2, 4.8/1.5),
                             constrained_layout=True)
        vmin = min(np.min(A1), np.min(A2))
        vmax = max(np.max(A1), np.max(A2))

    sns.heatmap(A1,
                cmap='viridis',
                annot=annot,
                fmt='.2f',
                cbar=False if A2 is not None else True,
                vmin=vmin,
                vmax=vmax,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot_kws={'size': annot_size},
                ax=axes[0])
    if titles is not None:
        axes[0].set_title(titles[0], fontsize=fontsize)

    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
    axes[0].tick_params(axis='x', labelsize=labelsize)
    axes[0].tick_params(axis='y', labelsize=labelsize)

    if A2 is not None:
        sns.heatmap(A2, cmap='viridis',
                    annot=annot,
                    fmt='.2f',
                    cbar=True,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    annot_kws={'size': 14},
                    ax=axes[1])

        if titles is not None:
            axes[1].set_title(titles[1], fontsize=fontsize)

        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
        axes[1].tick_params(axis='x', labelsize=labelsize)
        axes[1].tick_params(axis='y', labelsize=labelsize)

    cbar = axes[-1].collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    plot_save(fig,
              save_name='A_matrix_'+save_name if save_name else None,
              folder_name=folder_name)


def plot_contour_and_samples(target: Target,
                             dims: list[int] | None = None,
                             samples: Array | None = None,
                             levels=30,
                             grid_size=200,
                             range_vals=(-1.5, 1.5),
                             fontsize: int = 14,
                             tick_size: int = 14,
                             alpha: float = 0.5,
                             save_name: str | None = None,
                             folder_name: str | None = None,
                             ):
    """
    Plot unique pairwise marginal log-pdf contours of a GMM,
    and overlay samples.

    Args:
        target: GMM target.
        dims: list of dimension indices to visualize.
        samples: (N, D) array of samples to overlay (optional).
    """

    means, covs, weights = target.gmm_params()

    if dims is None:
        dims = list(range(target.gmm.shape[0]))
    else:
        dims = [d for d in dims if d < target.gmm.shape[0]]

    n_dims = len(dims)
    n_plots = n_dims * (n_dims-1) / 2

    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))
    fig_size = (6.4*(1.5**(n_cols-1)), 4.8*(1.5**(n_rows-1)))

    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=fig_size,
                             sharey=True,
                             sharex=True,
                             constrained_layout=True)

    axes = np.array(axes).flatten()

    dim_combinations = list(combinations(dims, 2))

    for idx, (i, j) in enumerate(dim_combinations):

        ax = axes[idx]

        # marginalized GMM
        means_sub = means[:, [i, j]]
        covs_sub = covs[:, [i, j]][:, :, [i, j]]
        gmm_sub = GaussianMixture(means_sub, covs_sub, weights)

        x = np.linspace(range_vals[0], range_vals[1], grid_size)
        y = np.linspace(range_vals[0], range_vals[1], grid_size)
        xx, yy = np.meshgrid(x, y)

        grid = jnp.column_stack([xx.ravel(), yy.ravel()])

        log_probs = gmm_sub.log_prob(grid)
        log_probs = log_probs.reshape(xx.shape)

        ax.contourf(xx, yy, log_probs, levels=levels, cmap='viridis')

        if samples is not None:
            samples_proj = samples[:, [i, j]]
            ax.plot(samples_proj[:, 0], samples_proj[:, 1], 'o', alpha=alpha)

        ax.set_xlabel(fr'$x_{{z_{{{i}}}}}$', fontsize=fontsize)
        ax.set_ylabel(fr'$x_{{z_{{{j}}}}}$', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)

        ax.set_xlim(*range_vals)
        ax.set_ylim(*range_vals)

    for k in range(int(n_plots), len(axes)):
        axes[k].axis('off')

    plot_save(fig,
              save_name='GMM_contour_'+save_name if save_name else None,
              folder_name=folder_name)


def dihedral_map():
    mymap = np.array([[0.9, 0.9, 0.9],
                       [0.85, 0.85, 0.85],
                       [0.8, 0.8, 0.8],
                       [0.75, 0.75, 0.75],
                       [0.7, 0.7, 0.7],
                       [0.65, 0.65, 0.65],
                       [0.6, 0.6, 0.6],
                       [0.55, 0.55, 0.55],
                       [0.5, 0.5, 0.5],
                       [0.45, 0.45, 0.45],
                       [0.4, 0.4, 0.4],
                       [0.35, 0.35, 0.35],
                       [0.3, 0.3, 0.3],
                       [0.25, 0.25, 0.25],
                       [0.2, 0.2, 0.2],
                       [0.15, 0.15, 0.15],
                       [0.1, 0.1, 0.1],
                       [0.05, 0.05, 0.05],
                       [0, 0, 0]])
    newcmp = clr.ListedColormap(mymap)
    return newcmp


def annotate_alanine_histrogram(axis=None):
    if axis is None:
        target = plt
        target.xlabel(r'$\phi$ in $\mathrm{deg}$')
        target.ylabel(r'$\psi$ in $\mathrm{deg}$')
        target.xlim([-180, 180])
        target.ylim([-180, 180])
    else:
        target = axis
        target.set_xlabel(r'$\phi$ in $\mathrm{deg}$')
        target.set_ylabel(r'$\psi$ in $\mathrm{deg}$')
        target.set_xlim([-180, 180])
        target.set_ylim([-180, 180])

    target.text(-155, 90, '$C5$', fontsize=18)
    target.text(-70, 90, '$C7eq$', fontsize=18)
    target.text(145, 90, '$C5$', fontsize=18)
    target.text(-155, -150, '$C5$', fontsize=18)
    target.text(-70, -150, '$C7eq$', fontsize=18)
    target.text(145, -150, '$C5$', fontsize=18)
    target.text(-170, -90, r'$\alpha_R$"', fontsize=18)
    target.text(140, -90, r'$\alpha_R$"', fontsize=18)
    target.text(-70, -90, r'$\alpha_R$', fontsize=18)
    target.text(70, 0, r'$\alpha_L$', fontsize=18)
    target.plot([-180, 13], [74, 74], 'k', linewidth=0.5)
    target.plot([128, 180], [74, 74], 'k', linewidth=0.5)
    target.plot([13, 13], [-180, 180], 'k', linewidth=0.5)
    target.plot([128, 128], [-180, 180], 'k', linewidth=0.5)
    target.plot([-180, 13], [-125, -125], 'k', linewidth=0.5)
    target.plot([128, 180], [-125, -125], 'k', linewidth=0.5)
    target.plot([-134, -134], [-125, 74], 'k', linewidth=0.5)
    target.plot([-110, -110], [-180, -125], 'k', linewidth=0.5)
    target.plot([-110, -110], [74, 180], 'k', linewidth=0.5)


def plot_rama(list_angles: list[Array],
              titles: Sequence[str] | None = None,
              cmap: clr.Colormap | None = None,
              save_name: str | None = None,
              folder_name: str | None = None,
              ):
    """Plot 2D Ramachandran density histogram for alanine from the dihedral 
    angles.
    
    Args:
    list_angles: list of angles in form of [N_samples x N_angles]."""
    if cmap is None:
        cmap = dihedral_map()

    n_plots = len(list_angles)
    fig, axs = plt.subplots(ncols=n_plots, figsize=(6.4 * n_plots, 4.8),
                            constrained_layout=True)

    images = []
    for i in range(n_plots):
        h, x_edges, y_edges  = jnp.histogram2d(
            list_angles[i][:, 0], list_angles[i][:, 1], bins=60, density=True)
        h_masked = np.where(h == 0, np.nan, h)
        x, y = np.meshgrid(x_edges, y_edges)
        images.append(axs[i].pcolormesh(x,y,h_masked.T, cmap=cmap))
        if titles:
            axs[i].set_title(titles[i])
        annotate_alanine_histrogram(axs[i])

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Density')

    plot_save(fig,
              save_name='Ala_rama_'+save_name if save_name else None,
              folder_name=folder_name)


def plot_1d_dihedrals(list_angles: Sequence[Array],
                      labels: Sequence[str] | None = None,
                      colors: Sequence[ColorType] | None = None,
                      linestyle: Sequence[LineStyleType] | None = None,
                      bins: int = 60,
                      fontsize: int = 14,
                      save_name: str | None = None,
                      folder_name: str | None = None,
                      ):
    '''Plot and save 1D histogram spline for alanine dipeptide dihedral
    angles with mean and standard deviation.

    Args:
    list_angles: list of angles in form of [N_samples x N_angles] with the 
    first entry being the reference.
    '''

    scale_x = 2.0
    scale_y = 0.9

    fig, (ax1, ax2)= plt.subplots(1, 2,
                                  figsize=[6.4 * scale_x, 4.8 * scale_y],
                                  constrained_layout=True)
    if labels is None:
        labels = ['Prediction', 'Target']

    if colors is None:
        colors = ['tab:blue', 'tab:orange']

    if linestyle is None:
        linestyle = ['-', (0, (4, 3))]

    n_models = len(list_angles)
    #put reference last in list_angles
    list_angles.append(list_angles.pop(0))

    for i in range(n_models):

        angles_phi = list_angles[i][:, 0]
        angles_psi = list_angles[i][:, 1]

        h_phi, x_bins  = jnp.histogram(angles_phi, bins=bins, density=True)
        h_psi, _  = jnp.histogram(angles_psi, bins=bins, density=True)
        width = x_bins[1]-x_bins[0]
        bin_center = x_bins + width/2

        ax1.plot(bin_center[:-1], h_phi, label=labels[i], color=colors[i],
                                        linestyle=linestyle[i], linewidth=2.0)
        ax2.plot(bin_center[:-1], h_psi, label=labels[i], color=colors[i],
                                        linestyle=linestyle[i], linewidth=2.0)

    ax1.set_xlabel(r'$\phi$ in $\mathrm{deg}$', fontsize=fontsize)
    ax1.set_ylabel('Density',   fontsize=fontsize)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)

    ax2.set_xlabel(r'$\psi$ in $\mathrm{deg}$', fontsize=fontsize)
    ax2.set_ylabel('Density', fontsize=fontsize)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)


    handles, lbs = plt.gca().get_legend_handles_labels()

    n_labels = len(labels)
    order = np.arange(n_labels, dtype=int)

    order -= 1
    order[0] = n_labels-1
    ax2.legend([handles[idx] for idx in order],[lbs[idx] for idx in order],
                                loc='upper left',
                                fontsize=fontsize)

    plot_save(fig,
              save_name='1D_dihedral_'+save_name if save_name else None,
              folder_name=folder_name)
