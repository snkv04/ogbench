import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags, logging
from tqdm import trange
import imageio.v2 as imageio

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle
from ogbench.manipspace.oracles.hierarchical.cube_hierarchical import CubeHierarchicalOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_bool('save_first_episode_video', False, 'If true, save a video of the first episode.')
flags.DEFINE_float('noise', 0.1, 'Action noise level.')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing level for PlanOracle.')
flags.DEFINE_float('min_norm', 0.4, 'Minimum action norm for MarkovOracle.')
flags.DEFINE_float('p_random_action', 0, 'Probability of selecting a random action.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Number of episodes.')
flags.DEFINE_bool('hierarchical', False, 'If true and dataset_type is noisy, use hierarchical oracle for cube tasks.')


def main(_):
    assert FLAGS.dataset_type in ['play', 'noisy']
    # 'play': Use a non-Markovian oracle (PlanOracle) that follows a pre-computed plan.
    # 'noisy': Use a Markovian, closed-loop oracle (MarkovOracle) with Gaussian action noise.

    # Set default save_path based on env_name and dataset_type if not provided.
    if FLAGS.save_path is None:
        dataset_suffix = 'hrl' if (FLAGS.hierarchical and FLAGS.dataset_type == 'noisy') else FLAGS.dataset_type
        FLAGS.save_path = f'.ogbench/data/{FLAGS.env_name}-{dataset_suffix}.npz'

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
    )

    # Initialize oracles.
    oracle_type = 'plan' if FLAGS.dataset_type == 'play' else 'markov'
    has_button_states = hasattr(env.unwrapped, '_cur_button_states')
    if 'cube' in FLAGS.env_name:
        if oracle_type == 'markov':
            if FLAGS.hierarchical:
                agents = {
                    'cube': CubeHierarchicalOracle(env=env, min_norm=FLAGS.min_norm),
                }
            else:
                agents = {
                    'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                }
        else:
            logging.info("LANDED HERE")
            agents = {
                'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif 'scene' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm, max_step=100),
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'drawer': DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'window': WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'button': ButtonPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'drawer': DrawerPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'window': WindowPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif 'puzzle' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True),
            }
        else:
            agents = {
                'button': ButtonPlanOracle(
                    env=env,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                    gripper_always_closed=True,
                ),
            }

    # Collect data.
    dataset = defaultdict(list)
    episode_frames = []
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        # Have an additional while loop to handle rare cases with undesirable states (for the Scene environment).
        while True:
            ob, info = env.reset()

            if ep_idx == 0 and FLAGS.save_first_episode_video:
                episode_frames = [env.render()]
            # Set the cube stacking probability for this episode.
            if 'single' in FLAGS.env_name:
                p_stack = 0.0
            elif 'double' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif 'triple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            elif 'quadruple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.1, 0.5)
            elif 'octuple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.35)
            else:
                p_stack = 0.5

            if oracle_type == 'markov':
                # Set the action noise level for this episode.
                xi = np.random.uniform(0, FLAGS.noise)

            agent = agents[info['privileged/target_task']]
            agent.reset(ob, info)

            done = False
            step = 0
            ep_qpos = []

            while not done:
                if np.random.rand() < FLAGS.p_random_action:
                    # Sample a random action.
                    action = env.action_space.sample()
                else:
                    # Get an action from the oracle.
                    action = agent.select_action(ob, info)
                    # logging.info(f'action type = {type(action)}')
                    # logging.info(f'action shape = {action.shape}')
                    # logging.info(f'action = {action}')
                    action = np.array(action)
                    if oracle_type == 'markov':
                        # Add Gaussian noise to the action.
                        action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if agent.done:
                    # Set a new task when the current task is done.
                    agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent = agents[agent_info['privileged/target_task']]
                    # logging.info('task was done, calling agent.reset()')
                    # logging.info(f'type(agent) = {type(agent)}')
                    agent.reset(agent_ob, agent_info)
                    # logging.info('finished calling agent.reset()')
                    # logging.info(f'agent._plan = {agent._plan}')

                dataset['observations'].append(ob)
                dataset['actions'].append(action)
                dataset['terminals'].append(done)
                dataset['qpos'].append(info['prev_qpos'])
                dataset['qvel'].append(info['prev_qvel'])
                if has_button_states:
                    dataset['button_states'].append(info['prev_button_states'])
                ep_qpos.append(info['prev_qpos'])

                if ep_idx == 0 and FLAGS.save_first_episode_video:
                    episode_frames.append(env.render())

                ob = next_ob
                step += 1

            if 'scene' in FLAGS.env_name:
                # Perform health check. We want to ensure that the cube is always visible unless it's in the drawer.
                # Otherwise, the test-time goal images may become ambiguous.
                is_healthy = True
                ep_qpos = np.array(ep_qpos)
                block_xyzs = ep_qpos[:, 14:17]
                if (block_xyzs[:, 1] >= 0.29).any():
                    is_healthy = False  # Block goes too far right.
                if ((block_xyzs[:, 1] <= -0.3) & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))).any():
                    is_healthy = False  # Block goes too far left, without being in the drawer.

                if is_healthy:
                    break
                else:
                    # Remove the last episode and retry.
                    print('Unhealthy episode, retrying...', flush=True)
                    for k in dataset.keys():
                        dataset[k] = dataset[k][:-step]
            else:
                break

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

        if (
            FLAGS.save_first_episode_video
            and FLAGS.save_path is not None
            and ep_idx == 0
            and episode_frames
        ):
            save_base = pathlib.Path(FLAGS.save_path)
            video_path = save_base.parent / f'{save_base.stem}_episode0.mp4'
            video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 30
            with imageio.get_writer(
                video_path.as_posix(),
                fps=fps,
                codec='libx264',
                quality=8,
                macro_block_size=None,
            ) as writer:
                for frame in episode_frames:
                    writer.append_data(frame)

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path.replace('.npz', '-train.npz')
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        elif k == 'button_states':
            dtype = np.int64
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
