import wandb
import sys
from main_run import main, all_data_loader
from get_sample import generate_sample_images, sample_data_loader
import os
import fcntl

def get_next_run_id(counter_file="run_id_counter.txt"):
    with open(counter_file, "a+") as f:
        f.seek(0)
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            content = f.read().strip()
            run_id = int(content) + 1 if content else 1
            f.seek(0)
            f.truncate()
            f.write(str(run_id))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return run_id

def run_sweep():
    run = wandb.init()
    try:
        lr_dis = run.config.lr_dis
        lr_gen = run.config.lr_gen
        lr_rec = run.config.lr_rec
        loss_type = run.config.loss_type

        run_id = get_next_run_id()
        wandb.config.run_id = run_id  # Save in wandb config

        train_loader, test_loader, imp_loader = all_data_loader()
        main(
            train_loader,
            test_loader,
            imp_loader,
            lr_dis=lr_dis,
            lr_gen=lr_gen,
            lr_rec=lr_rec,
            run_id=run_id,
            loss_type=loss_type  # Pass loss_type
        )

        sample_loader = sample_data_loader()
        generate_sample_images(sample_loader, run_id=run_id)
    except Exception as e:
        print(f"Error during sweep run {run.id}: {e}")
        wandb.log({"error": str(e)})
        run.finish(exit_code=1)
    else:
        run.finish()

if __name__ == '__main__':
    run_sweep()