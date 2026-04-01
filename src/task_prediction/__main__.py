from dotenv import load_dotenv
import asyncio
import logging
import nats
import typer
from typing import Annotated, Optional

from .configs import AppSettings
from .core.factories import create_system, create_sinks, get_logger
from .runners.server import ServerRunner

load_dotenv()

# Initialize Typer CLI
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Task Intent Prediction Engine"
)

@app.command()
def serve(
    model_dir: Annotated[Optional[str], typer.Option(
        "--model-dir", "-m", help="Override the XGBoost model directory path."
    )] = None,
    host: Annotated[Optional[str], typer.Option(
        "--nats-host", "-n", help="Override the NATS broker URL."
    )] = None,
):
    """
    Start the Real-time Prediction Server.
    
    Loads configuration from environment variables (TASK_PRED__) or .env file.
    """
    # Load and Validate Configuration
    # Overrides can be passed via kwargs to model_validate or env vars
    overrides = {}
    if model_dir:
        overrides["model"] = {"model_dir": model_dir}
    if host:
        overrides["nats_host"] = host

    try:
        settings = AppSettings(**overrides)
    except Exception as e:
        typer.secho(f"Configuration Error: {e}", fg="red", err=True)
        raise typer.Exit(1)

    # Setup Logging
    logging.getLogger("nats").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("tsfresh").setLevel(logging.ERROR)
    logger = get_logger(settings.logging)
    
    logger.info(f"Starting Prediction Engine v{settings.__version__}")
    logger.info(f"Subscribing to NATS: {settings.nats_host}")
    logger.info(f"Sampling Interval: {settings.sampling_interval_ms}ms")

    # A single NATS client is shared for subscribing and for the NATSSink
    nc = nats.NATS()

    # Build the Runner
    runner = ServerRunner(
        system=create_system(settings),
        sinks=create_sinks(settings, nc=nc),
        nc=nc,
        nats_host=settings.nats_host,
        sampling_interval_ms=settings.sampling_interval_ms
    )

    # Execute Async Event Loop
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Closing...")
    except Exception as e:
        logger.critical(f"System failure: {e}", exc_info=True)
        raise typer.Exit(1)

@app.command()
def version():
    """Print the version and exit."""
    settings = AppSettings()
    typer.echo(f"Task Prediction Engine version: {settings.__version__}")

if __name__ == "__main__":
    app()