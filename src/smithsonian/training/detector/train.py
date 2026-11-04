"""Train RF-DETR 2XL on target-centered crops with ignore-region supervision.

Uses the recorded Fall recipe and the Spring single-class initialization.
See NOTICE.md for source revisions and credit. CUDA is required for training.
"""

from smithsonian.training.detector.loss import (
    patch_criterion_for_ignore,
    seed_class_head_from_one_class,
    pollen_row_for,
)
from smithsonian.io.artifacts import sha256 as sha
from smithsonian.preprocessing.detector_crops.common import write


from smithsonian.configuration.options import ConfigArgumentParser, save_effective_config

import importlib.metadata
import json
import math
import os
import time
from pathlib import Path


def main():
    """Parse configuration and run this workflow explicitly.

    Returns:
        None; writes only the outputs requested by the command.
    """
    p = ConfigArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", required=True, help="COCO dataset produced by prepare_ignore_regions.py"
    )
    p.add_argument("--init", required=True, help="pretrained .pth (one-class Spring detector)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1e-5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument(
        "--pollen-row",
        type=int,
        default=None,
        help="head row the one-class init fires on; read from the dataset summary when present",
    )
    p.add_argument(
        "--time-budget-min",
        type=float,
        default=None,
        help="chunked mode: save resume.ckpt and stop cleanly after this many minutes of training",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="chunked mode: Lightning checkpoint written by a previous chunk",
    )
    p.add_argument(
        "--init-sha256", required=True, help="Expected SHA-256 of the initialization checkpoint"
    )
    a = p.parse_args()
    from smithsonian.io.artifacts import verify

    verify(Path(a.init), a.init_sha256)
    if a.resume:
        previous = Path(a.out) / "effective_config.json"
        if not previous.is_file():
            raise ValueError("Legacy resume record: use a fresh run directory")
        recorded = json.loads(previous.read_text())
        for key in ("dataset", "init", "init_sha256", "steps", "lr", "lr_encoder", "seed"):
            if recorded["settings"].get(key) != getattr(a, key):
                raise ValueError(f"Incompatible resume setting: {key}")
        from smithsonian.configuration.options import package_fingerprints

        if recorded["package_sources"] != package_fingerprints():
            raise ValueError("Package source changed since this run began")
    if not a.resume and Path(a.out).exists():
        raise FileExistsError("Choose a new detector training output directory")
    save_effective_config(
        a, Path(a.out) / ("resume_config.json" if a.resume else "effective_config.json")
    )

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from rfdetr.config import TrainConfig
    from rfdetr_plus.models.detection import RFDETR2XLargeConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
    from rfdetr.training.callbacks import RFDETREMACallback
    from rfdetr._namespace import build_namespace
    import rfdetr.datasets.coco as coco

    assert importlib.metadata.version("rfdetr") == "1.6.4" and torch.cuda.is_available()
    data, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        assert Path(a.resume).exists() and json.loads((out / "status.json").read_text())[
            "status"
        ] in ("paused", "training")
    train_coco = json.loads((data / "train/_annotations.coco.json").read_text())
    cats = sorted(train_coco["categories"], key=lambda c: c["id"])
    real = [c for c in cats if c["name"] != "__ignore__"]
    assert len(real) == 28, len(real)
    # rfdetr 1.6.4 trains with remap_category_ids=True: contiguous labels in sorted id order,
    # so __ignore__ (the max id) becomes label 28 and the 28 real classes are 0..27.
    ignore_label = len(real) if len(real) != len(cats) else None
    if len(cats) != 29 or cats[-1]["name"] != "__ignore__":
        raise ValueError("Expected 28 labeled categories followed by __ignore__")
    n_train = len(train_coco["images"])
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    pl.seed_everything(a.seed, workers=True)

    def resize_only(scales, *, square, max_size=None):  # same as team: no secondary random crop
        """Build the fixed detector resize transform without a secondary random crop.

        Args:
            scales: Expected single resolution, [1000].
            square: Must be True for the recorded detector input.
            max_size: Unused compatibility argument from RF-DETR.

        Returns:
            One Albumentations resize configuration.
        """
        assert square and scales == [1000]
        return [{"Resize": {"height": 1000, "width": 1000}}]

    coco._build_train_resize_config = resize_only

    steps = 2 if a.smoke else a.steps
    epochs = 1 if a.smoke else math.ceil(steps / max(1, n_train // 16)) + 1
    if a.time_budget_min:
        # a resumed epoch restarts from its first batch, so chunking can consume a few extra epochs;
        # max_steps still caps estimated_stepping_batches, so the cosine schedule is unchanged.
        epochs += 6
    aug = {"HorizontalFlip": {"p": 0.5}}
    aug.update({"VerticalFlip": {"p": 0.5}, "RandomRotate90": {"p": 0.5}})
    mc = RFDETR2XLargeConfig(pretrain_weights=a.init, num_classes=28, resolution=1000)
    settings = dict(
        dataset_dir=str(data),
        output_dir=str(out),
        epochs=epochs,
        batch_size=1,
        grad_accum_steps=16,
        num_workers=2,
        multi_scale=False,
        expanded_scales=False,
        use_ema=True,
        early_stopping=False,
        tensorboard=False,
        wandb=False,
        run_test=False,
        seed=a.seed,
        lr=a.lr,
        lr_encoder=a.lr_encoder,
        lr_scheduler="cosine",
        compute_val_loss=False,
        log_per_class_metrics=False,
        progress_bar=None,
        save_dataset_grids=False,
        aug_config=aug,
    )
    tc = TrainConfig(**settings)
    module = RFDETRModelModule(mc, tc)
    assert module.model.class_embed.out_features == 29, module.model.class_embed.out_features
    init_head = seed_class_head_from_one_class(
        module.model, a.init, pollen_row_for(data, a.pollen_row)
    )
    if ignore_label is not None:
        patch_criterion_for_ignore(module.criterion, ignore_label)
    dm = RFDETRDataModule(mc, tc)
    contract = dict(
        experiment="A4_ignore_springinit",
        smoke=a.smoke,
        max_optimizer_steps=steps,
        init_checkpoint=a.init,
        init_sha256=sha(a.init),
        init_head=init_head,
        ignore_label=ignore_label,
        train_images=n_train,
        train_coco_sha256=sha(data / "train/_annotations.coco.json"),
        training_settings=settings,
        model_config=mc.model_dump(),
        torch=torch.__version__,
        rfdetr=importlib.metadata.version("rfdetr"),
        gpu=torch.cuda.get_device_name(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        selection="fixed final EMA, no validation-based selection",
        chunked=bool(a.time_budget_min),
        chunks=[],
    )
    if a.resume:  # keep the original contract; append this chunk
        contract = json.loads((out / "run_contract.json").read_text())
    contract["chunks"].append(
        dict(
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            resume=a.resume,
            time_budget_min=a.time_budget_min,
            epochs_cap=epochs,
        )
    )
    write(out / "run_contract.json", contract)
    started = time.time()
    paused = {"flag": False}

    class TimeBudget(Callback):
        """Chunked mode: save a full Lightning checkpoint (model, optimizer, LR schedule, EMA state)
        and stop before the partition wall limit. Saved mid-step, before any callback swaps weights."""

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            """Record training progress and, when explicitly configured, save a resumable segment.

            Args:
                trainer: Lightning trainer.
                module: RF-DETR training module.
                outputs: Current batch outputs.
                batch: Current image batch.
                batch_idx: Index of the batch in this epoch.

            Returns:
                None; updates the run status or resume artifact.
            """
            if (
                a.time_budget_min
                and not paused["flag"]
                and trainer.global_step < steps
                and time.time() - started > a.time_budget_min * 60
                and (batch_idx + 1) % 16 == 0
            ):
                trainer.save_checkpoint(str(out / "resume.ckpt"))
                write(
                    out / "status.json",
                    dict(
                        status="paused",
                        optimizer_step=trainer.global_step,
                        max_steps=steps,
                        epoch=trainer.current_epoch,
                        chunk_elapsed_seconds=time.time() - started,
                    ),
                )
                paused["flag"] = True
                trainer.should_stop = True

    class Progress(Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            """Record training progress and, when explicitly configured, save a resumable segment.

            Args:
                trainer: Lightning trainer.
                module: RF-DETR training module.
                outputs: Current batch outputs.
                batch: Current image batch.
                batch_idx: Index of the batch in this epoch.

            Returns:
                None; updates the run status or resume artifact.
            """
            if batch_idx % 100 == 0:
                write(
                    out / "progress.json",
                    dict(
                        status="training",
                        optimizer_step=trainer.global_step,
                        max_steps=steps,
                        epoch=trainer.current_epoch,
                        elapsed_seconds=time.time() - started,
                        ignore_hits_last_batch=getattr(module.criterion, "_ignore_hits", None),
                    ),
                )

        def on_train_epoch_end(self, trainer, module):
            """Report completed epoch progress.

            Args:
                trainer: Lightning trainer.
                module: RF-DETR training module.

            Returns:
                None; prints a JSON progress record.
            """
            print(
                json.dumps(
                    dict(
                        epoch=trainer.current_epoch,
                        optimizer_step=trainer.global_step,
                        elapsed_seconds=round(time.time() - started),
                    )
                ),
                flush=True,
            )

    callbacks = [
        RFDETREMACallback(
            decay=tc.ema_decay, tau=tc.ema_tau, update_interval_steps=tc.ema_update_interval
        ),
        TimeBudget(),
        Progress(),
    ]
    if not a.smoke and not a.time_budget_min:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(out),
                filename="resume",
                every_n_epochs=5,
                save_top_k=1,
                save_last=False,
                enable_version_counter=False,
                auto_insert_metric_name=False,
                save_on_train_epoch_end=True,
            )
        )
    trainer = build_trainer(
        tc,
        mc,
        logger=CSVLogger(save_dir=str(out / "logs"), name="", version=""),
        callbacks=callbacks,
        max_steps=steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_checkpointing=not a.smoke,
    )
    write(out / "status.json", {"status": "training", "resumed_from": a.resume})
    trainer.fit(module, dm, ckpt_path=a.resume)
    if paused["flag"]:
        print(json.dumps(json.loads((out / "status.json").read_text())), flush=True)
        return
    assert trainer.global_step == steps, trainer.global_step
    args = build_namespace(mc, tc)
    args.class_names = [c["name"] for c in real]
    state = {k: v.detach().cpu() for k, v in module.model.state_dict().items()}
    torch.save(
        {
            "model": state,
            "args": args,
            "epoch": trainer.current_epoch,
            "optimizer_steps": trainer.global_step,
            "selection": "fixed_final_ema",
        },
        out / "checkpoint_final_ema.pth",
    )
    result = dict(
        status="completed",
        optimizer_steps=trainer.global_step,
        epochs_started=trainer.current_epoch,
        elapsed_seconds=time.time() - started,
        checkpoint_sha256=sha(out / "checkpoint_final_ema.pth"),
        test_used=False,
    )
    write(out / "status.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
