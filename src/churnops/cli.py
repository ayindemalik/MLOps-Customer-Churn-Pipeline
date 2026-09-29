"""Command line: one command per pipeline stage. DVC calls these."""

import json

import typer

from churnops.config import load_params

app = typer.Typer(help="Customer churn MLOps pipeline", no_args_is_help=True)


@app.command() # this decorator makes the function a CLI command
def fetch():
    """Download the raw Telco churn CSV."""
    from churnops.data import download
    path = download(load_params()["data"]["url"])
    typer.echo(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB)")


@app.command()
def prepare():
    """Clean, validate against the schema, split into train/val/test."""
    from churnops.data import prepare as run
    typer.echo(f"Rows per split: {run(load_params())}")


@app.command()
def train():
    """Train the model in params.yaml and log the run to MLflow."""
    from churnops.train import train as run
    result = run(load_params())
    typer.echo(json.dumps(result, indent=2))


@app.command()
def evaluate():
    """Score on the test set, apply the quality gate, maybe promote."""
    from churnops.evaluate import evaluate as run
    scores = run(load_params())
    typer.echo(json.dumps(scores, indent=2))
    if not scores["passed_gate"]:
        typer.echo("QUALITY GATE FAILED: model not promoted.", err=True)
        raise typer.Exit(code=1)


@app.command()
def drift():
    """Compare a simulated 'next month' batch with the training data."""
    import pandas as pd

    from churnops.config import PROCESSED, REPORTS
    from churnops.drift import report, simulate_next_month

    reference = pd.read_parquet(PROCESSED / "train.parquet")
    current = simulate_next_month(pd.read_parquet(PROCESSED / "test.parquet"))
    table = report(reference, current)
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "drift.csv", index=False)
    typer.echo(table.to_string(index=False))
    if (table["status"] == "drift").any():
        typer.echo("\nDRIFT DETECTED: retraining recommended.")


if __name__ == "__main__":
    app()
