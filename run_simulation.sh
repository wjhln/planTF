clear
export HYDRA_FULL_ERROR=1
export NUPLAN_MAPS_ROOT='/home/wang/Project/nuplan/dateset/maps'
export NUPLAN_DATA_ROOT='/home/wang/Project/nuplan/dateset/'
export NUPLAN_EXP_ROOT='./exp'

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"

    # 'scenario_filter=all_scenarios',
    # # 'scenario_filter.scenario_types=[starting_left_turn]',
    # # 'scenario_filter.limit_total_scenarios=1',
    # 'scenario_filter.scenario_tokens=["c742dfbe4e4c5b60"]',

python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    scenario_builder=nuplan_mini \
    scenario_filter=all_scenarios \
    scenario_filter.scenario_tokens=["c742dfbe4e4c5b60"] \
    verbose=true \
    planner.imitation_planner.planner_ckpt="/home/wang/Project/nuplan/planTF/checkpoints/planTF.ckpt"