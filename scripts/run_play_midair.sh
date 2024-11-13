# cd ..

python source/standalone/workflows/rl_games/play.py \
    --task Isaac-Repose-Cube-Shadow-Direct-Face-Down-Rotate-Lift-v0 \
    --num_envs 1  \
    --checkpoint '/home/yibo/workspace/syb/IsaacLab/logs/rl_games/shadow_hand_rotate_lift/on_table/nn/shadow_hand_rotate_lift.pth'


# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information 