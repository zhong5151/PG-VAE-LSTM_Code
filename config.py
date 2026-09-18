from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


MODEL_ALIASES = {
    "lstm": "lstm",
    "ed-lstm": "ed_lstm",
    "ed_lstm": "ed_lstm",
    "vae-lstm": "vae_lstm",
    "vae_lstm": "vae_lstm",
    "pg-vae-lstm": "pg_vae_lstm",
    "pg_vae_lstm": "pg_vae_lstm",
}


def str2bool(value: str | bool) -> bool:
    """Parse common command-line Boolean values safely."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid Boolean value: {value}")


def build_parser(description: str = "PG-VAE-LSTM") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--output_path", type=str, default="outputs/pg_vae_lstm")

    parser.add_argument("--input_path", type=str, default="data/processed")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--x_file", type=str, default="X.npy")
    parser.add_argument("--y_file", type=str, default="y.npy")
    parser.add_argument("--site_file", type=str, default="site_id.npy")
    parser.add_argument("--energy_file", type=str, default="energy_available_wm2.npy")
    parser.add_argument("--water_file", type=str, default="water_available_mm_day.npy")
    parser.add_argument("--seq_len", type=int, default=365)
    parser.add_argument("--input_size", type=int, default=39)
    parser.add_argument("--normalize_inputs", type=str2bool, default=True)
    parser.add_argument("--normalize_target", type=str2bool, default=True)
    parser.add_argument("--scaler_file", type=str, default=None)

    # Model settings
    parser.add_argument(
        "--modelname",
        type=str,
        default="PG-VAE-LSTM",
        choices=["LSTM", "ED-LSTM", "VAE-LSTM", "PG-VAE-LSTM"],
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.15)

    # Loss settings
    parser.add_argument("--beta_kl", type=float, default=0.001)
    parser.add_argument("--gamma_phy", type=float, default=0.1)
    parser.add_argument("--free_bits", type=float, default=0.0)
    parser.add_argument("--physical_equal_weight_average", type=str2bool, default=True)
    parser.add_argument("--energy_enabled", type=str2bool, default=True)
    parser.add_argument("--water_enabled", type=str2bool, default=True)
    parser.add_argument("--nonnegative_enabled", type=str2bool, default=True)
    parser.add_argument("--energy_mode", type=str, default="flux", choices=["flux", "et_equivalent"])
    parser.add_argument("--latent_heat_j_per_kg", type=float, default=2.45e6)

    # Training settings
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--niter", type=int, default=500, help="Training steps per epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.0)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_total_loss",
        choices=["val_total_loss", "val_rmse", "val_kge"],
    )
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=1000)

    # Evaluation settings
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--min_site_samples", type=int, default=2)
    parser.add_argument("--save_predictions", type=str2bool, default=True)

    return parser


def namespace_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert flat command-line arguments to the internal configuration structure."""
    model_key = MODEL_ALIASES[args.modelname.lower()]
    output_dir = Path(args.output_path)
    scaler_file = args.scaler_file or str(output_dir / "scaler.npz")

    cfg: Dict[str, Any] = {
        "experiment": {
            "seed": args.seed,
            "device": args.device,
            "output_dir": str(output_dir),
        },
        "data": {
            "root": args.input_path,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "x_file": args.x_file,
            "y_file": args.y_file,
            "site_file": args.site_file,
            "energy_file": args.energy_file,
            "water_file": args.water_file,
            "seq_len": args.seq_len,
            "input_size": args.input_size,
            "normalize_inputs": args.normalize_inputs,
            "normalize_target": args.normalize_target,
            "scaler_file": scaler_file,
        },
        "model": {
            "name": model_key,
            "hidden_size": args.hidden_size,
            "latent_size": args.latent_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout_rate,
        },
        "loss": {
            "beta_kl": args.beta_kl,
            "gamma_phy": args.gamma_phy,
            "free_bits": args.free_bits,
            "physical_equal_weight_average": args.physical_equal_weight_average,
            "energy_term": {
                "enabled": args.energy_enabled,
                "mode": args.energy_mode,
                "latent_heat_j_per_kg": args.latent_heat_j_per_kg,
            },
            "water_term": {"enabled": args.water_enabled},
            "nonnegative_term": {"enabled": args.nonnegative_enabled},
        },
        "training": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "epochs": args.epochs,
            "steps_per_epoch": args.niter,
            "num_workers": args.num_workers,
            "weight_decay": args.weight_decay,
            "deterministic": args.deterministic,
            "gradient_clip_norm": args.gradient_clip_norm,
            "monitor": args.monitor,
            "maximize_monitor": args.monitor == "val_kge",
            "min_delta": args.min_delta,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "evaluation": {
            "batch_size": args.eval_batch_size,
            "num_workers": args.eval_num_workers,
            "min_site_samples": args.min_site_samples,
            "save_predictions": args.save_predictions,
        },
    }
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate the most important configuration values before an experiment starts."""
    if cfg["data"]["seq_len"] <= 0 or cfg["data"]["input_size"] <= 0:
        raise ValueError("seq_len and input_size must be positive integers")
    if cfg["model"]["hidden_size"] <= 0 or cfg["model"]["num_layers"] <= 0:
        raise ValueError("hidden_size and num_layers must be positive integers")
    if cfg["model"]["name"] in {"vae_lstm", "pg_vae_lstm"} and cfg["model"]["latent_size"] <= 0:
        raise ValueError("latent_size must be positive for VAE-based models")
    if cfg["training"]["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if cfg["training"]["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if cfg["loss"]["beta_kl"] < 0 or cfg["loss"]["gamma_phy"] < 0:
        raise ValueError("beta_kl and gamma_phy must be non-negative")


def get_args(description: str = "PG-VAE-LSTM", argv: list[str] | None = None) -> Dict[str, Any]:
    """Parse command-line arguments and return the internal configuration dictionary."""
    parser = build_parser(description)
    args = parser.parse_args(argv)
    return namespace_to_config(args)
