import subprocess

import click


def parse_multi_vars(ctx, param, value, sep=","):
    try:
        return [v for v in value.split(sep)]
    except ValueError:
        raise click.BadParameter(
            f"must be a space separated list of parameters (specified: {value})."
        )


@click.command()
@click.argument("job_name")
@click.option("--phase", default="train")
@click.option("--config_names", callback=parse_multi_vars, default="exp001")
@click.option("--folds", callback=parse_multi_vars, default="0,1,2,3,4")
@click.option("--seeds", callback=parse_multi_vars, default="42")
@click.option("--env", default="local")
@click.option("--dry_run", is_flag=True)
@click.option("--infer_batch_size", default=32)
@click.option("--no_eval", is_flag=True)
def run_experiments(
    job_name,
    phase,
    config_names,
    folds,
    seeds,
    env,
    dry_run,
    infer_batch_size,
    no_eval,
):
    for config_name in config_names:
        for fold in folds:
            for seed in seeds:
                cmd = f"python -m run.{job_name} --config-name={config_name} phase={phase} job_name={job_name} fold={fold} seed={seed} env={env} infer.batch_size={infer_batch_size} no_eval={no_eval}"
                print(cmd)
                if not dry_run:
                    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    run_experiments()
