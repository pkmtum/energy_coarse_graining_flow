"""Energy-based coarse-graining for flows: Gaussian Mixture Example."""
import jax
from jax import tree, random, numpy as jnp

import matplotlib.pyplot as plt

import equinox as eqx
import paramax
import optax

from ecg.targets.gmm import GMM
from ecg.train.losses import EnergyLoss
from ecg.models import initialize_model
from ecg.train.loops import fit_to_key_beta_based_loss, TemperingScheme
from ecg.utils import Dimensions, create_output_dirs

jax.config.update('jax_enable_x64', True)

folder_name = 'GMM_10D/'
print(folder_name or '')

seeds = {'key': 1, 'gmm': 6, 'cond': 1}
dimensions = Dimensions(x_dim=20, z_dim=10)
num_components = 3
cluster_std = 0.1
cond_std = 0.1

# Train input:
epochs = 1000
num_samples = 10000
plot_training = True
initial_train = 20

# Model input:
Phi_init = jnp.identity(dimensions.x_dim) * 0.2

layers=8
nn_depth=2
nn_width=40
knots=8
interval = 4.
cond_nn_depth = 2
cond_nn_width = 40

# Optimizer input:
learning_rate = 1e-3
optimizer = optax.adam(learning_rate = learning_rate)

# Tempering input:
start_beta = 0.001
target_beta = 1.0

temp_dict = {
             'method': 'adaptive',
             'delta_kl': 0.1,
             'delta_beta': 0.02,
             'num_samples': 10000,
            }

# Plotting input:
plotted_dims = [0, 1, 2]
annotate_matrix = True if dimensions.x_dim <= 10 else False

# Save path inputs:
save_params_path = f'outputs/trained_parameter/{folder_name}'
load_params_path = None

create_output_dirs(
    folder_name,
    save_params_path,
    subdirs=['contours', 'losses', 'marginals', 'transformations']
)

key = random.key(seeds['key'])

target = GMM(seeds, dimensions, num_components, cluster_std, cond_std)

key, init_model_key, init_eval_key = random.split(key, 3)

model = initialize_model(dimensions,
                         Phi_init,
                         key=init_model_key,
                         flow_layers=layers,
                         nn_depth=nn_depth,
                         nn_width=nn_width,
                         knots=knots,
                         interval=interval,
                         load_params_path=load_params_path,
                        )


params, static = eqx.partition(
    model,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )

param_count = sum(x.size for x in tree.leaves(params))
print('Number of parameters:', param_count)

target.plot(model,
            init_eval_key,
            dims=plotted_dims,
            annotate_matrix=annotate_matrix,
            save_name='init',
            folder_name=folder_name)


kl_loss = EnergyLoss(energy=target.energy,
                     num_samples=num_samples,
                     weight_samples=temp_dict['num_samples'])

temp_dict['get_log_weights'] = kl_loss.compute_log_weights


Tempering = TemperingScheme(start_beta,
                            target_beta,
                            **temp_dict)

losses = []
grad_norms = []

opt_state = optimizer.init(params)

print('energy training')
beta = start_beta

for i, _ in enumerate(Tempering.loop()):

    key, train_key, adap_key, eval_key = random.split(key, 4)

    # Train longer at initial beta
    if initial_train and i < 1:
        current_epochs = epochs * initial_train
    else:
        current_epochs = epochs

    # Train with reverse KL
    model, infos = fit_to_key_beta_based_loss(
                    train_key,
                    model,
                    beta,
                    loss_fn=kl_loss,
                    steps=current_epochs,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    return_best=True,
                    )

    losses.extend(infos.losses)
    grad_norms.extend(infos.grad_norm)

    if plot_training:
        target.plot(model,
                    eval_key,
                    dims=plotted_dims,
                    annotate_matrix=annotate_matrix,
                    save_name=f'train_{beta:.5f}',
                    folder_name=folder_name)

    # Continue from last optimizer step
    opt_state = infos.opt_state

    # Tempering Scheme to update beta
    beta = Tempering.step(model, adap_key)

    if folder_name is not None:
        model.save(save_params_path+'intermediates/'+
            f'model_params_{beta:.5f}beta_{len(losses)}epochs.eqx')


if folder_name is not None:
    model.save(save_params_path+f'model_params_{beta:.5f}beta.eqx')

plt.figure()
plt.plot(losses)
plt.yscale('symlog')
plt.xlabel('Epoch')
plt.ylabel('Loss')
if folder_name is not None:
    plt.savefig(f'outputs/figures/{folder_name}losses/Loss.png')

plt.figure()
plt.plot(grad_norms)
plt.yscale('symlog')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
if folder_name is not None:
    plt.savefig(f'outputs/figures/{folder_name}losses/Grad_norm.png')

key, final_eval_key = random.split(key, 2)
target.plot(model,
            final_eval_key,
            dims=plotted_dims,
            annotate_matrix=annotate_matrix,
            save_name='final',
            folder_name=folder_name)
