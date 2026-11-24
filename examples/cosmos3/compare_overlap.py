"""Export baseline/FASTER host timelines, measured overlap and comparison plots.

No processes are launched here. Inputs are the files produced by separate
server/client commands. All interval arithmetic uses unions, not nested sums.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative timing interval")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def overlap(left, right):
    return duration((max(a, c), min(b, d)) for a, b in union(left) for c, d in union(right) if min(b, d) > max(a, c))


def intervals(events, stage):
    return [(row["start_s"], row["end_s"]) for row in events if row["stage"] == stage]


def load_run(client_dir, server_dir):
    summary = json.loads((client_dir / "summary.json").read_text())
    requests = read_jsonl(client_dir / "request_timings.jsonl")
    events = read_jsonl(client_dir / "client_timeline.jsonl")
    event_index = defaultdict(list)
    for event in events:
        event_index[event.get("trace_id")].append(event)
    workers = read_jsonl(server_dir / "worker_timeline.jsonl")
    worker_index = {(int(row["pid"]), int(row["trace_id"])): row for row in workers}
    windows = []
    for request in requests:
        trace = request.get("trace_id")
        if not trace:
            raise ValueError("Missing trace_id; rerun with the overlap instrumentation enabled")
        selected = [dict(event, source="client") for event in event_index[trace]]
        server = request.get("server_timeline") or {}
        domain = request["clock_domain"]
        if server.get("clock_domain") != domain:
            raise ValueError("Server/client have different monotonic clock domains; do not overlay their timestamps")
        selected += [dict(event, trace_id=trace, source="server", clock_domain=domain) for event in server["events"]]
        for metric in request.get("worker_metrics", []):
            key = (int(metric["worker_pid"]), int(metric["worker_trace_id"]))
            worker = worker_index[key]
            if worker["clock_domain"] != domain:
                raise ValueError("Worker/client clock domains differ")
            selected += [
                dict(event, trace_id=trace, source="worker", clock_domain=domain, stage="worker." + event["stage"])
                for event in worker["events"]
            ]
        compute = intervals(selected, "worker.forward")
        if not compute:
            raise ValueError(f"No measured worker.forward for {trace}")
        execute = intervals(selected, "client.execute_action")
        wait = intervals(selected, "client.wait_action")
        render = intervals(selected, "client.render_observation")
        request_interval = intervals(selected, "client.request")
        start = min(event["start_s"] for event in selected)
        end = max(event["end_s"] for event in selected)
        active = [event for event in selected if event["stage"] != "client.render_observation"]
        active_start = min(event["start_s"] for event in active)
        no_render_s = end - active_start - overlap([(active_start, end)], render)
        first_action = min((a for a, _ in execute), default=None)
        intersection = overlap(compute, execute)
        row = {
            "trace_id": trace,
            "request": request["request"],
            "task_id": request.get("task_id"),
            "episode": request.get("episode"),
            "warmup": request.get("warmup", False),
            "actions_executed": sum(
                len(event.get("env_slots", [0])) for event in selected if event["stage"] == "client.execute_action"
            ),
            "expected_actions": int(summary["rendering"]["execution_window"]) * request["batch_size"],
            "start_s": start,
            "end_s": end,
            "window_wall_s": end - start,
            "window_elapsed_no_render_s": no_render_s,
            "render_excluded_s": duration(render),
            "worker_forward_s": duration(compute),
            "client_execute_s": duration(execute),
            "client_wait_s": duration(wait),
            "client_wait_final_s": duration(intervals(selected, "client.wait_final")),
            "render_rpc_s": duration(render),
            "http_roundtrip_s": duration(request_interval),
            "worker_execute_overlap_s": intersection,
            "overlap_fraction_of_worker": intersection / duration(compute) if duration(compute) else 0,
            "observation_to_first_action_s": first_action - start if first_action is not None else None,
            "request_to_first_action_s": first_action - request_interval[0][0] if first_action is not None else None,
        }
        windows.append(row)
        for event in selected:
            event.update(
                task_id=request.get("task_id"),
                episode=request.get("episode"),
                warmup=request.get("warmup", False),
                request=request["request"],
            )
        # One canonical event list, including server and worker intervals for
        # exactly the requests made by this client; warmup RPCs aren't mixed in.
        request["_events"] = selected
    return {
        "summary": summary,
        "requests": requests,
        "windows": windows,
        "events": [event for request in requests for event in request["_events"]],
    }


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            for row in rows
        )


def aggregate(run):
    # Exclude timing warmup AND incomplete final windows from the matched
    # 16-action window comparison. Keep every event in the exported raw files.
    horizon = int(run["summary"]["rendering"]["execution_window"])
    windows = [
        row for row in run["windows"] if not row["warmup"] and row["actions_executed"] == row["expected_actions"]
    ]
    if not windows:
        raise ValueError("No complete measured execution windows")
    fields = [
        "window_wall_s",
        "window_elapsed_no_render_s",
        "worker_forward_s",
        "client_execute_s",
        "client_wait_s",
        "client_wait_final_s",
        "render_rpc_s",
        "http_roundtrip_s",
        "worker_execute_overlap_s",
        "overlap_fraction_of_worker",
        "request_to_first_action_s",
        "observation_to_first_action_s",
    ]
    result = {"complete_measured_windows": len(windows), "execution_window": horizon}
    for field in fields:
        values = [row[field] for row in windows if row[field] is not None]
        result[field] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    result["accuracy"] = run["summary"]["accuracy"]["overall"]
    count = sum(row["actions_executed"] for row in windows)
    elapsed = sum(row["window_elapsed_no_render_s"] for row in windows)
    result["primary_no_render"] = {
        "executed_actions": count,
        "elapsed_s": elapsed,
        "actions_per_s": count / elapsed if elapsed > 0 else None,
    }
    result["evaluation_wall"] = run["summary"]["throughput"]["evaluation_wall"]
    return result


def task_comparison(runs):
    """Preserve task identity; different trajectory lengths aren't paired windows."""
    result = []
    tasks = runs["baseline"]["summary"]["selected_task_ids"]
    fields = [
        "window_wall_s",
        "window_elapsed_no_render_s",
        "worker_forward_s",
        "client_execute_s",
        "client_wait_s",
        "render_rpc_s",
        "worker_execute_overlap_s",
        "request_to_first_action_s",
    ]
    for task_id in tasks:
        row = {"task_id": task_id}
        for name, run in runs.items():
            task = next(task for task in run["summary"]["task_results"] if task["task_id"] == task_id)
            windows = [
                window
                for window in run["windows"]
                if window["task_id"] == task_id
                and not window["warmup"]
                and window["actions_executed"] == window["expected_actions"]
            ]
            row.update(
                {
                    f"{name}_episodes": task["episodes"],
                    f"{name}_successes": task["successes"],
                    f"{name}_executed_actions": sum(episode["executed_actions"] for episode in task["episode_results"]),
                    f"{name}_episode_wall_s": sum(episode["elapsed_s"] for episode in task["episode_results"]),
                    f"{name}_complete_measured_windows": len(windows),
                }
            )
            for field in fields:
                row[f"{name}_{field}_mean"] = statistics.mean(window[field] for window in windows) if windows else None
        a, b = row["baseline_window_elapsed_no_render_s_mean"], row["faster_window_elapsed_no_render_s_mean"]
        row["mean_window_speedup"] = a / b if a is not None and b else None
        a, b = row["baseline_request_to_first_action_s_mean"], row["faster_request_to_first_action_s_mean"]
        row["first_action_wait_reduction_percent"] = 100 * (1 - b / a) if a and b is not None else None
        result.append(row)
    return result


def plot_task_comparison(tasks, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    fields = [
        ("request_to_first_action_s", "Request to first executed action"),
        ("worker_execute_overlap_s", "Model / control overlap"),
        ("window_elapsed_no_render_s", "Execution window (render excluded)"),
    ]
    for ax, (field, label) in zip(axes, fields):
        for name, offset, color in [("baseline", -0.2, "#7570b3"), ("faster", 0.2, "#1b9e77")]:
            ax.bar(
                [index + offset for index in range(len(tasks))],
                [
                    float("nan") if row.get(f"{name}_{field}_mean") is None else row[f"{name}_{field}_mean"]
                    for row in tasks
                ],
                width=0.38,
                label=name,
                color=color,
            )
        ax.set_xticks(range(len(tasks)), [str(row["task_id"]) for row in tasks])
        ax.set_xlabel("LIBERO task ID")
        ax.set_ylabel("Seconds (mean complete non-warmup window)")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    figure.tight_layout()
    for extension in ["png", "svg"]:
        figure.savefig(output / f"task_timing_comparison.{extension}", dpi=180)
    plt.close(figure)


def plot(runs, output, full_episode=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [
        ("worker.forward", "Model worker forward", "#d95f02"),
        ("worker.transformer", "Transformer host spans", "#e6ab02"),
        ("server.policy", "Policy frontend", "#7570b3"),
        ("client.request", "Client HTTP / stream", "#999999"),
        ("client.wait_action", "Client waits for action", "#e7298a"),
        ("client.execute_action", "Client physics / control", "#1b9e77"),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    for ax, (name, run) in zip(axes, runs.items()):
        horizon = run["summary"]["rendering"]["execution_window"]
        window = next(
            row for row in run["windows"] if not row["warmup"] and row["actions_executed"] == row["expected_actions"]
        )
        rows = [
            event
            for event in run["events"]
            if (event.get("task_id"), event.get("episode")) == (window["task_id"], window["episode"])
        ]
        if not full_episode:
            rows = [event for event in rows if event["trace_id"] == window["trace_id"]]
        render = union(intervals(rows, "client.render_observation"))
        rows = [event for event in rows if event["stage"] != "client.render_observation"]
        origin = min(event["start_s"] for event in rows)

        def render_free_time(value):
            return value - origin - overlap([(origin, max(value, origin))], render)

        scale = 1 if full_episode else 1000
        for lane, (stage, label, color) in enumerate(stages):
            spans = [
                (render_free_time(a) * scale, (render_free_time(b) - render_free_time(a)) * scale)
                for a, b in intervals(rows, stage)
            ]
            ax.broken_barh(spans, (lane - 0.32, 0.64), facecolors=color)
        for event in rows:
            if event["stage"] == "server.action_ready":
                ax.plot(render_free_time(event["start_s"]) * scale, 0, "|", color="black", markersize=11)
        ax.set_yticks(range(len(stages)), [item[1] for item in stages])
        ax.invert_yaxis()
        ax.set_xlabel(
            f"{'Seconds' if full_episode else 'Milliseconds'} of shared server/client time (camera / render RPC excluded)"
        )
        ax.set_title(
            f"{name}: task {window['task_id']}, episode {window['episode']}; rendering excluded"
            if full_episode
            else f"{name}: first complete measured {horizon}-action window; "
            f"worker/control overlap={window['worker_execute_overlap_s'] * 1000:.2f} ms"
        )
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_xlim(0, max(ax.get_xlim()[1] for ax in axes))
    axes[1].set_xlim(axes[0].get_xlim())
    figure.suptitle("Server/client overlap without rendering; same clock transform applied to both lanes")
    figure.tight_layout()
    for extension in ["png", "svg"]:
        figure.savefig(output / f"{'episode_overlap' if full_episode else 'overlap'}.{extension}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-client", type=Path, required=True)
    parser.add_argument("--baseline-server", type=Path, required=True)
    parser.add_argument("--faster-client", type=Path, required=True)
    parser.add_argument("--faster-server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        "baseline": load_run(args.baseline_client, args.baseline_server),
        "faster": load_run(args.faster_client, args.faster_server),
    }
    # Check the main experiment controls instead of comparing unlike windows.
    for key in ["task_suite", "selected_task_ids", "num_trials_per_task", "action_space", "action_dim"]:
        if runs["baseline"]["summary"][key] != runs["faster"]["summary"][key]:
            raise ValueError(f"Comparison configuration mismatch: {key}")
    for key in ["seed", "max_steps", "action_horizon", "num_envs", "env_image_size", "camera"]:
        left = runs["baseline"]["summary"]["run_config"].get(key)
        right = runs["faster"]["summary"]["run_config"].get(key)
        if left != right:
            raise ValueError(f"Comparison run configuration mismatch: {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {name: aggregate(run) for name, run in runs.items()}
    base, faster = report["baseline"], report["faster"]
    if base["execution_window"] != faster["execution_window"]:
        raise ValueError("Execution windows differ")
    report["comparison"] = {
        "mean_window_speedup": base["window_elapsed_no_render_s"]["mean"]
        / faster["window_elapsed_no_render_s"]["mean"],
        "median_window_speedup": base["window_elapsed_no_render_s"]["median"]
        / faster["window_elapsed_no_render_s"]["median"],
        "primary_no_render_throughput_speedup": faster["primary_no_render"]["actions_per_s"]
        / base["primary_no_render"]["actions_per_s"],
        "mean_first_action_wait_reduction_percent": 100
        * (1 - faster["request_to_first_action_s"]["mean"] / base["request_to_first_action_s"]["mean"]),
        "mean_action_wait_reduction_percent": 100
        * (1 - faster["client_wait_s"]["mean"] / base["client_wait_s"]["mean"]),
        "raw_evaluation_wall_speedup_including_render": base["evaluation_wall"]["elapsed_s"]
        / faster["evaluation_wall"]["elapsed_s"],
        "timing_scope": "host forward/control intervals; device substage spans are asynchronous unless synchronized",
        "statistics_scope": "complete non-warmup windows, camera/render RPC excluded; raw episode wall times are diagnostic only",
        "interpretation": "Measured overlap and wall-time changes are reported separately; no speedup is assumed.",
    }
    tasks = task_comparison(runs)
    report["by_task"] = tasks
    comparable = [row for row in tasks if row["mean_window_speedup"] is not None]
    if comparable:
        report["comparison"]["task_balanced_mean_window_speedup"] = statistics.mean(
            row["baseline_window_elapsed_no_render_s_mean"] for row in comparable
        ) / statistics.mean(row["faster_window_elapsed_no_render_s_mean"] for row in comparable)
    successful = [
        row
        for row in tasks
        if row["baseline_successes"] == row["baseline_episodes"] > 0
        and row["faster_successes"] == row["faster_episodes"] > 0
    ]
    report["comparison"]["paired_successful_task_ids"] = [row["task_id"] for row in successful]
    if successful:
        report["comparison"]["raw_paired_successful_episode_wall_speedup_including_render"] = sum(
            row["baseline_episode_wall_s"] for row in successful
        ) / sum(row["faster_episode_wall_s"] for row in successful)
    report["comparison"]["accuracy_scope"] = (
        "All task outcomes are retained; overall wall time includes unsuccessful episodes. "
        "The paired-success metric uses only tasks completed by both profiles."
    )
    for name, run in runs.items():
        write_csv(args.output / f"{name}_windows.csv", run["windows"])
        write_csv(args.output / f"{name}_events.csv", run["events"])
        (args.output / f"{name}_events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in run["events"]))
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(
        args.output / "comparison.csv", [{"metric": key, "value": value} for key, value in report["comparison"].items()]
    )
    write_csv(args.output / "task_comparison.csv", tasks)
    plot(runs, args.output)
    plot(runs, args.output, full_episode=True)
    plot_task_comparison(tasks, args.output)
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
