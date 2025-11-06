"""Energy-based coarse-graining for flows: Double Well Example."""
import jax
from jax import tree, random, numpy as jnp

import matplotlib.pyplot as plt

import equinox as eqx
import paramax
import optax

from ecg.targets.dw import DoubleWell
from ecg.train.losses import EnergyLoss
from ecg.models import initialize_model
from ecg.train.loops import fit_to_key_beta_based_loss, TemperingScheme
from ecg.utils import Dimensions, create_output_dirs

jax.config.update('jax_enable_x64', True)

folder_name = 'DW/'
print(folder_name or '')

seed = 4
dimensions = Dimensions(x_dim=2, z_dim=1)

# Train input:
epochs = 100
num_samples = 500
plot_training = True
initial_train = 50

# Model input:
Phi_init = jnp.array([[0, 1.0],[0.,  2.]])

layers=6
nn_depth=2
nn_width=32
knots=8
interval = 5.
cond_nn_depth = 2
cond_nn_width = 32

# Optimizer input:
learning_rate = 1e-3
optimizer = optax.adam(learning_rate = learning_rate)

# Tempering input:
start_beta = 0.01
target_beta = 1.0

temp_dict = {
             'method': 'adaptive',
             'delta_kl': 0.1,
             'delta_beta': 0.05,
             'num_samples': 1000,
            }

# Save path inputs:
save_params_path = f'outputs/trained_parameter/{folder_name}'
load_params_path = None
create_output_dirs(
    folder_name,
    save_params_path,
    subdirs=['energy_marginals', 'histograms', 'kl', 'losses']
)
key = random.key(seed)
target = DoubleWell()
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
            start_beta,
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
phi_params = []

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
                    return_bij_params=True,
                    )

    losses.extend(infos.losses)
    grad_norms.extend(infos.grad_norm)
    phi_params.extend(infos.bij_params)

    if plot_training:
        target.plot(model,
                    eval_key,
                    beta,
                    bij_params=phi_params,
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
            target_beta,
            bij_params=phi_params,
            save_name='final',
            folder_name=folder_name)
inv_matrix = model.inv_matrix

print(
    'Learned Transformation based on A inverse:\n'
    f'z = ${inv_matrix[0][0]:.2f} * x_1 '
    f'{"+" if inv_matrix[0][1] >= 0 else "-"} '
    f'{abs(inv_matrix[0][1]):.2f} * x_2\n'
    f'X = ${inv_matrix[1][0]:.2f} * x_1 '
    f'{"+" if inv_matrix[1][1] >= 0 else "-"} '
    f'{abs(inv_matrix[1][1]):.2f} * x_2\n'
    )
