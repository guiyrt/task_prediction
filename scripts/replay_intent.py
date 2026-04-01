import asyncio
import argparse
import logging
import pyarrow.parquet as pq
import nats

from aware_protos.zhaw.protobuf import task_prediction_pb2, aircraft_attention_target_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def replay_task_prediction(nc: nats.NATS, filepath: str, speed: float = 1.0):
    """Replays a Task Prediction Parquet file to NATS."""
    logger.info(f"Loading Task Prediction Parquet: {filepath}")
    table = pq.read_table(filepath)
    rows = table.to_pylist()
    
    if not rows:
        logger.warning("Task Parquet file is empty.")
        return

    logger.info(f"Replaying {len(rows)} Task Predictions at {speed}x speed...")
    
    proto = task_prediction_pb2.TaskPrediction()
    last_ts = rows[0]["timestamp"]

    for row in rows:
        # Calculate and wait the time delta (respecting speed modifier)
        current_ts = row["timestamp"]
        delta_sec = (current_ts - last_ts).total_seconds()
        if delta_sec > 0:
            await asyncio.sleep(delta_sec / speed)
        last_ts = current_ts

        # Inject current timestamp
        proto.Clear()
        proto.timestamp.GetCurrentTime()
        
        # Map Status Enum
        if row["status"] is not None:
            proto.status = getattr(task_prediction_pb2.TaskPrediction, f"TASK_PRED_STATUS_{row['status']}", 0)

        # Map Active Probabilities
        if row["is_active"] is not None:
            proto.is_active = row["is_active"]
        if row["active_proba"] is not None:
            proto.active_proba = row["active_proba"]

        # Map Predicted Task Enum
        if row["pred_task"] is not None:
            proto.pred_task = getattr(task_prediction_pb2.TaskPrediction, f"TASK_TYPE_{row['pred_task']}", 0)

        # Map Task Probabilities Dictionary
        if row["task_probas"] is not None:
            for task_name, proba in row["task_probas"]:
                enum_val = getattr(task_prediction_pb2.TaskPrediction, f"TASK_TYPE_{task_name}", 0)
                proto.task_probas[enum_val] = proba

        await nc.publish("intent.task_prediction", proto.SerializeToString())

    logger.info("Task Prediction replay complete.")


async def replay_attention_target(nc: nats.NATS, filepath: str, speed: float = 1.0):
    """Replays an Aircraft Attention Target Parquet file to NATS."""
    logger.info(f"Loading Attention Target Parquet: {filepath}")
    table = pq.read_table(filepath)
    rows = table.to_pylist()

    if not rows:
        logger.warning("Attention Parquet file is empty.")
        return

    logger.info(f"Replaying {len(rows)} Attention Targets at {speed}x speed...")
    
    proto = aircraft_attention_target_pb2.AircraftAttentionTarget()
    last_ts = rows[0]["timestamp"]

    for row in rows:
        # Calculate and wait the time delta (respecting speed modifier)
        current_ts = row["timestamp"]
        delta_sec = (current_ts - last_ts).total_seconds()
        if delta_sec > 0:
            await asyncio.sleep(delta_sec / speed)
        last_ts = current_ts

        # Inject current timestamp
        proto.Clear()
        proto.timestamp.GetCurrentTime()

        callsigns = row["callsigns"] or []
        scores = row["scores"] or []
        indicators_list = row["indicators"] or []

        if callsigns:
            proto.primary_target_callsign = callsigns[0]

            for callsign, score, indicators in zip(callsigns, scores, indicators_list):
                target_msg = proto.targets.add()
                target_msg.callsign = callsign
                target_msg.score = score
                
                # Map Indicator Enums
                for ind_name in (indicators or []):
                    enum_val = getattr(aircraft_attention_target_pb2.AircraftAttentionTarget, f"INDICATOR_{ind_name}", 0)
                    target_msg.active_indicators.append(enum_val)

        await nc.publish("intent.aircraft_attention_target", proto.SerializeToString())

    logger.info("Attention Target replay complete.")


async def main():
    parser = argparse.ArgumentParser(description="Replay Parquet files to NATS as fresh events.")
    parser.add_argument("--task", type=str, help="Path to Task Prediction parquet file")
    parser.add_argument("--attention", type=str, help="Path to Attention Target parquet file")
    parser.add_argument("--nats", type=str, default="nats://localhost:4222", help="NATS URL")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g., 2.0 is twice as fast)")
    args = parser.parse_args()

    if not args.task and not args.attention:
        logger.error("You must provide at least one file to replay: --task or --attention")
        return

    # Connect to NATS
    nc = await nats.connect(args.nats)
    logger.info(f"Connected to NATS at {args.nats}")

    # Launch requested replay tasks concurrently
    tasks = []
    if args.task:
        tasks.append(replay_task_prediction(nc, args.task, args.speed))
    if args.attention:
        tasks.append(replay_attention_target(nc, args.attention, args.speed))

    await asyncio.gather(*tasks)

    # Graceful shutdown
    await nc.drain()
    logger.info("All replays finished. Disconnected from NATS.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nReplay manually stopped by user.")