export MUJOCO_GL=osmesa  # To enable headless rendering
python -m data_gen_scripts.generate_locomaze \
    --env_name=antmaze-medium-singletask-task1-v0 \
    --save_path=.ogbench/data/antmaze-medium-singletask-task1-v0.npz \
    --dataset_type=path \
    --num_episodes=10 \
    --max_episode_steps=200 \
    --restore_path=experts/ant \
    --restore_epoch=400000 \
    --save_first_episodes_videos=5
