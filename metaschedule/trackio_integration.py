from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import trackio
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.measure_callback import PyMeasureCallback
from tvm.s_tir.meta_schedule.task_scheduler import TaskScheduler


def _floats(xs: Any) -> list[float]:
    if xs is None:
        return []
    out: list[float] = []
    for x in xs:
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def _safe_str(x: Any, limit: int = 1000) -> str:
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        s = repr(x)
    return s[:limit]


def _candidate_trace_preview(candidate: Any, limit: int = 1200) -> str:
    sch = getattr(candidate, "sch", None)
    if sch is None:
        return ""
    trace_obj = getattr(sch, "trace", None)
    if callable(trace_obj):
        try:
            trace_obj = trace_obj()
        except Exception:
            trace_obj = None
    return _safe_str(trace_obj, limit=limit)


def _pretty_ms_value(x: Any) -> Any:
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    return x


def _pretty_ms_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Speed (GFLOPS)", "Latency (us)", "Weighted Latency (us)"]:
        out[col] = out[col].map(_pretty_ms_value)
    if "FLOP" in out.columns:
        out["FLOP"] = out["FLOP"].map(lambda x: "N/A" if pd.isna(x) else int(x))
    return out


def _is_scheduled_task(task: Any) -> bool:
    attrs = getattr(task.mod, "attrs", None)
    return bool(attrs and attrs.get("tir.is_scheduled", False))


def _record_to_row(record: Any, db_index: int) -> dict[str, Any]:
    run_secs = _floats(getattr(record, "run_secs", None))
    trace_text = str(getattr(record, "trace", ""))
    return {
        "db_index": db_index,
        "n_repeats": len(run_secs),
        "latency_median_us": float(np.median(run_secs) * 1e6) if run_secs else math.nan,
        "latency_mean_us": float(np.mean(run_secs) * 1e6) if run_secs else math.nan,
        "latency_min_us": float(np.min(run_secs) * 1e6) if run_secs else math.nan,
        "trace_preview": trace_text[:2000],
    }


@ms.derived_object
class LiveTrackioCallback(PyMeasureCallback):
    META_TABLE_COLUMNS = [
        "ID",
        "Name",
        "FLOP",
        "Weight",
        "Speed (GFLOPS)",
        "Latency (us)",
        "Weighted Latency (us)",
        "Trials",
        "Done",
    ]

    def __init__(
        self,
        *,
        task_names: list[str],
        task_weights: list[float],
        max_trials_global: int,
        summary_every: int = 25,
        top_k: int = 20,
    ) -> None:
        self.task_names = list(task_names)
        self.task_weights = [float(w) for w in task_weights]
        if len(self.task_names) != len(self.task_weights):
            raise ValueError(f"len(task_names)={len(self.task_names)} != len(task_weights)={len(self.task_weights)}")

        self.max_trials_global = int(max_trials_global)
        self.summary_every = max(1, int(summary_every))
        self.top_k = max(1, int(top_k))
        self.total_task_weight = float(sum(self.task_weights))

        self._lock = threading.Lock()
        self._start_time = time.perf_counter()

        self._step = 0
        self._success = 0
        self._build_fail = 0
        self._run_fail = 0
        self._best_global_us = math.inf

        n_tasks = len(self.task_names)
        self._best_per_task_us = {i: math.inf for i in range(n_tasks)}
        self._task_trials = {i: 0 for i in range(n_tasks)}

        self._successful_latencies_us: list[float] = []
        self._top_rows: list[dict[str, Any]] = []

    def _scheduler_tasks_unlocked(self, task_scheduler: Any | None) -> list[Any]:
        if task_scheduler is None:
            return []
        try:
            return list(getattr(task_scheduler, "tasks_", []))
        except Exception:
            return []

    def _meta_summary_unlocked(
        self,
        task_scheduler: Any | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        scheduler_tasks = self._scheduler_tasks_unlocked(task_scheduler)
        n_tasks = max(len(self.task_names), len(self.task_weights), len(scheduler_tasks))

        rows: list[dict[str, Any]] = []
        total_latency_us = 0.0
        meta_total_trials = 0
        covered_tasks = 0
        covered_weight = 0.0

        for i in range(n_tasks):
            task_rec = scheduler_tasks[i] if i < len(scheduler_tasks) else None

            task_name = self.task_names[i] if i < len(self.task_names) else f"task_{i}"
            if task_rec is not None:
                try:
                    ctx = getattr(task_rec, "ctx", None)
                    maybe_name = getattr(ctx, "task_name", None)
                    if maybe_name is not None:
                        task_name = str(maybe_name)
                except Exception:
                    pass

            task_weight = self.task_weights[i] if i < len(self.task_weights) else 1.0
            if task_rec is not None:
                try:
                    task_weight = float(task_rec.task_weight)
                except Exception:
                    pass

            flop = math.nan
            if task_rec is not None:
                try:
                    flop = float(task_rec.flop)
                except Exception:
                    pass

            done = ""
            if task_rec is not None:
                try:
                    done = "Y" if bool(task_rec.is_terminated) else ""
                except Exception:
                    pass

            trials = int(self._task_trials.get(i, 0))
            best_us = float(self._best_per_task_us.get(i, math.inf))

            if math.isfinite(best_us):
                latency_us = best_us
                speed_gflops = float(flop / latency_us / 1000.0) if math.isfinite(flop) else math.nan
                weighted_latency_us = float(latency_us * task_weight)

                total_latency_us += weighted_latency_us
                meta_total_trials += trials
                covered_tasks += 1
                covered_weight += task_weight
            else:
                latency_us = math.nan
                speed_gflops = math.nan
                weighted_latency_us = math.nan

            weight_value: int | float
            if float(task_weight).is_integer():
                weight_value = int(task_weight)
            else:
                weight_value = float(task_weight)

            rows.append(
                {
                    "ID": int(i),
                    "Name": task_name,
                    "FLOP": int(flop) if math.isfinite(flop) else math.nan,
                    "Weight": weight_value,
                    "Speed (GFLOPS)": speed_gflops,
                    "Latency (us)": latency_us,
                    "Weighted Latency (us)": weighted_latency_us,
                    "Trials": trials,
                    "Done": done,
                }
            )

        df = pd.DataFrame(rows, columns=self.META_TABLE_COLUMNS)
        totals = {
            "total_latency_us": float(total_latency_us),
            "meta_total_trials": int(meta_total_trials),
            "covered_tasks": int(covered_tasks),
            "covered_tasks_ratio": float(covered_tasks / max(1, n_tasks)),
            "covered_weight": float(covered_weight),
            "covered_weight_ratio": (
                float(covered_weight / self.total_task_weight) if self.total_task_weight > 0 else 0.0
            ),
        }
        return df, totals

    def _emit_periodic_summary_unlocked(self, step: int, summary_df: pd.DataFrame) -> None:
        if step != 1 and step % self.summary_every != 0:
            return

        trackio.log(
            {"meta_task_summary": trackio.Table(dataframe=_pretty_ms_table(summary_df))},
            step=step,
        )

        if self._top_rows:
            trackio.log(
                {"live_top_candidates": trackio.Table(dataframe=pd.DataFrame(self._top_rows))},
                step=step,
            )

        if self._successful_latencies_us:
            trackio.log(
                {
                    "live_latency_hist_us": trackio.Histogram(
                        self._successful_latencies_us,
                        num_bins=min(64, max(8, int(np.sqrt(len(self._successful_latencies_us))))),
                    )
                },
                step=step,
            )

    def current_step(self) -> int:
        with self._lock:
            return int(self._step)

    def current_totals(self, task_scheduler: Any | None = None) -> dict[str, Any]:
        with self._lock:
            _, totals = self._meta_summary_unlocked(task_scheduler)
            return dict(totals)

    def make_meta_task_summary_df(
        self,
        task_scheduler: Any | None = None,
        *,
        pretty: bool = False,
    ) -> pd.DataFrame:
        with self._lock:
            df, _ = self._meta_summary_unlocked(task_scheduler)
            return _pretty_ms_table(df) if pretty else df

    def apply(
        self,
        task_scheduler,
        task_id: int,
        measure_candidates,
        builder_results,
        runner_results,
    ) -> None:
        with self._lock:
            for candidate, builder_result, runner_result in zip(measure_candidates, builder_results, runner_results):
                self._step += 1
                step = self._step
                elapsed_sec = time.perf_counter() - self._start_time
                task_name = self.task_names[task_id] if 0 <= task_id < len(self.task_names) else f"task_{task_id}"

                self._task_trials[task_id] += 1

                build_error = _safe_str(getattr(builder_result, "error_msg", None), limit=500)
                run_error = _safe_str(getattr(runner_result, "error_msg", None), limit=500)
                run_secs = _floats(getattr(runner_result, "run_secs", None))

                metrics: dict[str, Any] = {
                    "task_id": int(task_id),
                    "trial": int(step),
                    "progress_pct": float(100.0 * step / max(1, self.max_trials_global)),
                    "elapsed_sec": float(elapsed_sec),
                    "trials_per_sec": float(step / max(elapsed_sec, 1e-9)),
                }

                if build_error:
                    self._build_fail += 1
                    summary_df, totals = self._meta_summary_unlocked(task_scheduler)
                    metrics.update(totals)
                    metrics.update(
                        {
                            "build_failed_total": int(self._build_fail),
                            "run_failed_total": int(self._run_fail),
                            "success_total": int(self._success),
                            "success_rate": float(self._success / step),
                            "build_failed": 1,
                            "run_failed": 0,
                            "succeeded": 0,
                        }
                    )
                    trackio.log(metrics, step=step)
                    self._emit_periodic_summary_unlocked(step, summary_df)
                    continue

                if run_error or not run_secs:
                    self._run_fail += 1
                    summary_df, totals = self._meta_summary_unlocked(task_scheduler)
                    metrics.update(totals)
                    metrics.update(
                        {
                            "build_failed_total": int(self._build_fail),
                            "run_failed_total": int(self._run_fail),
                            "success_total": int(self._success),
                            "success_rate": float(self._success / step),
                            "build_failed": 0,
                            "run_failed": 1,
                            "succeeded": 0,
                        }
                    )
                    trackio.log(metrics, step=step)
                    self._emit_periodic_summary_unlocked(step, summary_df)
                    continue

                latency_median_us = float(np.median(run_secs) * 1e6)
                latency_mean_us = float(np.mean(run_secs) * 1e6)
                latency_min_us = float(np.min(run_secs) * 1e6)

                self._success += 1
                self._successful_latencies_us.append(latency_median_us)
                self._best_global_us = min(self._best_global_us, latency_median_us)
                self._best_per_task_us[task_id] = min(
                    self._best_per_task_us[task_id],
                    latency_median_us,
                )

                task_best_latency_us = self._best_per_task_us[task_id]
                task_weight = self.task_weights[task_id]

                self._top_rows.append(
                    {
                        "trial": int(step),
                        "task_id": int(task_id),
                        "task_name": task_name,
                        "latency_median_us": latency_median_us,
                        "task_best_latency_us": task_best_latency_us,
                        "weighted_latency_us": task_best_latency_us * task_weight,
                        "n_repeats": int(len(run_secs)),
                        "trace_preview": _candidate_trace_preview(candidate),
                    }
                )
                self._top_rows.sort(key=lambda row: row["latency_median_us"])
                self._top_rows = self._top_rows[: self.top_k]

                summary_df, totals = self._meta_summary_unlocked(task_scheduler)
                metrics.update(totals)
                metrics.update(
                    {
                        "build_failed_total": int(self._build_fail),
                        "run_failed_total": int(self._run_fail),
                        "success_total": int(self._success),
                        "success_rate": float(self._success / step),
                        "build_failed": 0,
                        "run_failed": 0,
                        "succeeded": 1,
                        "latency_median_us": latency_median_us,
                        "latency_mean_us": latency_mean_us,
                        "latency_min_us": latency_min_us,
                        "best_latency_us": float(self._best_global_us),
                        "task_best_latency_us": float(task_best_latency_us),
                        "task_weighted_latency_us": float(latency_median_us * task_weight),
                        "task_best_weighted_latency_us": float(task_best_latency_us * task_weight),
                        "n_repeats": int(len(run_secs)),
                    }
                )
                trackio.log(metrics, step=step)
                self._emit_periodic_summary_unlocked(step, summary_df)

    def __str__(self) -> str:
        return "LiveTrackioCallback"


def tune_relax_to_trackio(
    mod,
    params,
    target,
    *,
    work_dir: str = "tuning_logs",
    project: str = "tvm-metaschedule",
    run_name: str = "run",
    max_trials_global: int = 2000,
    max_trials_per_task: int | None = None,
    num_trials_per_iter: int = 64,
    strategy: str = "evolutionary",
    task_scheduler: str | TaskScheduler = "gradient",
    cost_model: str = "xgb",
    space_id: str | None = None,
    auto_log_gpu: bool = False,
    summary_every: int = 25,
    top_k: int = 20,
    open_browser: bool = False,
    dashboard_host: str | None = None,
):
    trackio.init(
        project=project,
        name=run_name,
        space_id=space_id,
        auto_log_gpu=auto_log_gpu,
        config={
            "target": str(target),
            "work_dir": work_dir,
            "max_trials_global": max_trials_global,
            "max_trials_per_task": max_trials_per_task,
            "num_trials_per_iter": num_trials_per_iter,
            "strategy": strategy,
            "task_scheduler": str(task_scheduler),
            "cost_model": cost_model,
        },
    )

    app = trackio.show(
        project=project,
        open_browser=open_browser,
        block_thread=False,
        host=dashboard_host,
    )
    print(f"[Trackio] dashboard: {getattr(app, 'url', None)}", flush=True)
    share_url = getattr(app, "share_url", None)
    if share_url:
        print(f"[Trackio] share_url: {share_url}", flush=True)

    extracted_tasks = ms.relax_integration.extract_tasks(
        mod=mod,
        target=target,
        params=params,
    )
    extracted_tasks = [task for task in extracted_tasks if not _is_scheduled_task(task)]

    if not extracted_tasks:
        trackio.finish()
        raise ValueError("No unscheduled Relax tasks were extracted.")

    tasks, task_weights = ms.relax_integration.extracted_tasks_to_tune_contexts(
        extracted_tasks=extracted_tasks,
        work_dir=work_dir,
        strategy=strategy,
    )
    task_weights = [float(w) for w in task_weights]
    task_names = [task.task_name for task in extracted_tasks]

    live_cb = LiveTrackioCallback(
        task_names=task_names,
        task_weights=task_weights,
        max_trials_global=max_trials_global,
        summary_every=summary_every,
        top_k=top_k,
    )

    # step=0 から MetaSchedule 互換の表を出しておく
    trackio.log(
        {
            "num_extracted_tasks": len(extracted_tasks),
            **live_cb.current_totals(),
            "meta_task_summary": trackio.Table(dataframe=live_cb.make_meta_task_summary_df(pretty=True)),
        },
        step=0,
    )

    scheduler = task_scheduler if isinstance(task_scheduler, TaskScheduler) else TaskScheduler.create(task_scheduler)

    measure_callbacks = ms.MeasureCallback.create("default") + [live_cb]

    database = ms.tune_tasks(
        tasks=tasks,
        task_weights=task_weights,
        work_dir=work_dir,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder="local",
        runner="local",
        database="json",
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=scheduler,
    )

    # tuning 終了後に Done / FLOP を scheduler から取り直して確定版を出す
    final_step = live_cb.current_step()
    trackio.log(
        {
            **live_cb.current_totals(task_scheduler=scheduler),
            "meta_task_summary": trackio.Table(
                dataframe=live_cb.make_meta_task_summary_df(
                    task_scheduler=scheduler,
                    pretty=True,
                )
            ),
        },
        step=final_step,
    )

    records = list(database.get_all_tuning_records())
    rows = [_record_to_row(rec, i) for i, rec in enumerate(records)]
    df = pd.DataFrame(rows)

    if not df.empty:
        df["best_db_order_median_us"] = df["latency_median_us"].cummin()

        valid = df["latency_median_us"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(valid):
            trackio.log({"final_latency_hist_us": trackio.Histogram(valid)}, step=final_step)

        top50 = df.sort_values("latency_median_us", kind="stable").head(50)
        trackio.log(
            {
                "final_top50": trackio.Table(
                    dataframe=top50[["db_index", "latency_median_us", "n_repeats", "trace_preview"]]
                )
            },
            step=final_step,
        )
    else:
        trackio.log({"num_records": 0}, step=0)

    trackio.save(f"{work_dir}/*")
    trackio.finish()
    return database, df, app


if __name__ == "__main__":
    import torch
    import torchvision
    from tvm.relax.frontend.torch import from_exported_program

    model = torchvision.models.resnet18()
    model.eval()
    ep = torch.export.export(model, (torch.randn(1, 3, 224, 224),))
    mod = from_exported_program(ep)
    mod = tvm.transform.Sequential(
        [
            tvm.relax.transform.LegalizeOps(),
            tvm.relax.transform.AnnotateTIROpPattern(),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.FuseOps(),
            tvm.relax.transform.FuseTIR(),
        ]
    )(mod)
    print(mod.script())

    from datetime import datetime

    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    database, df, app = tune_relax_to_trackio(
        mod=mod,
        params=None,
        target=tvm.target.Target.from_device(tvm.cuda(0)),
        work_dir="tuning_logs",
        project="tvm-metaschedule",
        run_name=run_name,
        max_trials_global=20000,
        num_trials_per_iter=64,
        strategy="evolutionary",
        task_scheduler="gradient",
        cost_model="xgb",
        space_id=None,
        auto_log_gpu=False,
        summary_every=20,  # 20 trial ごとに table / hist を更新
        top_k=20,
        open_browser=False,  # ローカル GUI があるなら True でもよい
        dashboard_host=None,  # リモートなら "0.0.0.0"
    )

    print(df.head())
