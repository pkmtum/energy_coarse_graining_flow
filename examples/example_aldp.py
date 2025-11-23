"""Energy-based coarse-graining for flows: Alanine Dipeptide Example."""
import jax
from jax import tree, random, numpy as jnp

import matplotlib.pyplot as plt

import equinox as eqx
import paramax

from ecg.targets.aldp import AlanineDipeptide
from ecg.train.losses import EnergyLoss
from ecg.models import initialize_model
from ecg.train.loops import fit_to_key_beta_based_loss, TemperingScheme
from ecg.utils import Dimensions, init_optimizer, create_output_dirs

jax.config.update('jax_enable_x64', True)

folder_name = 'ALDP/'
print(folder_name or '')

seed = 4
# Fix coordinates to remove roto-translational invariance
fix_coord_idxs = jnp.array([9, 10, 11, 12, 13, 19], dtype=jnp.int32)
dimensions = Dimensions(x_dim=24, z_dim=15,
                        dofs=fix_coord_idxs)

temperature = 330 # Kelvin

# Train input:
epochs = 1000
num_samples = 10000
plot_training = True
initial_train = 15

# Model input:
# Give small initial bias towards z in transformation Phi_init to prevent
# singular matrix A_phi
Phi_init =  jnp.insert(jnp.eye(dimensions.num_atoms-2), 2,
                     jnp.eye(dimensions.num_atoms-2)[2], axis=0) * 0.25

init_z_bias_idx = jnp.array([15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 7, 8,
                    9, 10, 11, 12, 13, 14, 21, 22, 23], dtype=jnp.int32)
# Restrain CA_z coordinate to be negative
restrain_idxs = jnp.array([9], dtype=jnp.int32)

layers=8
nn_depth=4
nn_width=64
knots=8
interval = 4
cond_nn_depth = 8
cond_nn_width = 90

# Optimizer input:
learning_rate = 5e-4
optimizer = init_optimizer(learning_rate = learning_rate,
                           dynamic_grad_norm_factor=5.,
                           dynamic_grad_norm_window=50,
                           )

# Tempering input:
start_beta = 0.0001
target_beta = 1.0

temp_dict = {
             'method': 'adaptive',
             'delta_kl': 0.2,
             'delta_beta': 0.005,
             'num_samples': 20000,
            }

# Plotting input:
num_plot_samples = 480000

# Save path inputs:
save_params_path = f'outputs/trained_parameter/{folder_name}'
load_params_path = None

create_output_dirs(
    folder_name,
    save_params_path,
    subdirs=['dihedrals',
             'losses',
             'observables',
             'transformations']
)

key = random.key(seed)
target = AlanineDipeptide(temperature, fix_coord_idxs=fix_coord_idxs)
key, init_model_key, init_eval_key = random.split(key, 3)

model = initialize_model(dimensions,
                         Phi_init,
                         key=init_model_key,
                         flow_layers=layers,
                         nn_depth=nn_depth,
                         nn_width=nn_width,
                         cond_nn_depth=cond_nn_depth,
                         cond_nn_width=cond_nn_width,
                         knots=knots,
                         interval=interval,
                         load_params_path=load_params_path,
                         perm_idxs=init_z_bias_idx,
                         restrain_idxs=restrain_idxs,
                         per_atom=True,
                        )

params, static = eqx.partition(
    model,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )

params_count = sum(x.size for x in tree.leaves(params))
print('Number of parameters:', params_count)

target.plot(model,
            init_eval_key,
            start_beta,
            num_samples=num_plot_samples,
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
                    beta,
                    num_samples=num_plot_samples,
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

key, final_plot_key, final_eval_key = random.split(key, 3)
target.plot(model,
            final_plot_key,
            target_beta,
            num_samples=num_plot_samples,
            save_name='final',
            folder_name=folder_name)

target.evaluate(model,
                final_eval_key,
                target_beta,
                num_samples=num_plot_samples,
                save_name='final',
                folder_name=folder_name)
