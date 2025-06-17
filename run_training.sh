clear
export HYDRA_FULL_ERROR=1
export NUPLAN_MAPS_ROOT='/home/wang/Project/nuplan/dateset/maps'
export NUPLAN_DATA_ROOT='/home/wang/Project/nuplan/dateset/'
export NUPLAN_EXP_ROOT='./exp'

    # 'scenario_filter=all_scenarios',
    # # 'scenario_filter.scenario_types=[starting_left_turn]',
    # # 'scenario_filter.limit_total_scenarios=1',
    # 'scenario_filter.scenario_tokens=["c742dfbe4e4c5b60"]',
    
python run_training.py \
    py_func=train \
    +training=train_planTF \
    worker=single_machine_thread_pool \
    worker.max_workers=1 \
    cache.use_cache_without_dataset=false \
    data_loader.params.batch_size=1 \
    data_loader.params.num_workers=24 \
    lr=1e-3\
    epochs=1 \
    warmup_epochs=0 \
    weight_decay=0.0001 \
    lightning.trainer.params.val_check_interval=1.0 \
    wandb.mode=disabled