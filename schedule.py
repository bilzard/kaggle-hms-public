import subprocess

import click


def parse_multi_vars(ctx, param, value):
    """
    スペースで区切られた文字列をリストに変換する
    """
    try:
        return [v for v in value.split()]
    except ValueError:
        raise click.BadParameter(
            f"must be a space separated list of parameters (specified: {value})."
        )


@click.command()
@click.argument("job_name")
@click.option("--phase", default="train")
@click.option("--exp_names", callback=parse_multi_vars, default="exp001")
@click.option("--folds", callback=parse_multi_vars, default="0 1 2 3 4")
@click.option("--seeds", callback=parse_multi_vars, default="42")
@click.option("--env", default="local")
@click.option("--dry_run", is_flag=True)
@click.option("--checkpointing", default="false")
def run_experiments(
    job_name, phase, exp_names, folds, seeds, env, dry_run, checkpointing
):
    for exp_name in exp_names:
        for fold in folds:
            for seed in seeds:
                cmd = f"python -m run.{job_name} --config-name={exp_name} phase={phase} job_name={job_name} fold={fold} seed={seed} env={env} architecture.model.encoder.grad_checkpointing={checkpointing}"
                print(cmd)
                if not dry_run:
                    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    run_experiments()
