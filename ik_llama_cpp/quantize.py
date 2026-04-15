"""Quantize GGUF models using ik_llama.cpp's llama-quantize.

Supports IQ4_KT and other ik_llama.cpp-specific quantization formats.

Usage:
    # Quantize with imatrix (recommended for IQ quants)
    ik-llama-quantize model-bf16.gguf model-IQ4_KT.gguf IQ4_KT --imatrix model-imatrix.gguf

    # Quantize without imatrix
    ik-llama-quantize model-bf16.gguf model-IQ4_KT.gguf IQ4_KT

    # Download bf16 + imatrix from HuggingFace and quantize in one step
    ik-llama-quantize --hf-repo bartowski/google_gemma-4-E2B-it-GGUF --hf-quant IQ4_KT

    # As a Python module
    python -m ik_llama_cpp.quantize model-bf16.gguf model-IQ4_KT.gguf IQ4_KT
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# ik_llama.cpp-specific quant types (superset of upstream llama.cpp)
IK_QUANT_TYPES = [
    "IQ4_KT", "IQ3_KT", "IQ2_KT", "IQ1_KT",
    "IQ4_KS", "IQ4_KSS", "IQ3_KS",
    "Q4_K_M", "Q4_K_S", "Q4_K_L",
    "Q8_0", "Q6_K", "Q5_K_M", "Q3_K_M",
]


def find_quantize_bin() -> Path | None:
    """Find the llama-quantize binary bundled with this package or on PATH."""
    # 1. Check inside the installed package (ik_llama_cpp/bin/)
    pkg_bin = Path(__file__).parent / "bin"
    for name in ["llama-quantize.exe", "llama-quantize"]:
        candidate = pkg_bin / name
        if candidate.is_file():
            return candidate

    # 2. Check PATH
    which = shutil.which("llama-quantize")
    if which:
        return Path(which)

    # 3. Check common build dirs relative to source tree
    src_root = Path(__file__).resolve().parent.parent
    vendor_src = src_root / "vendor" / "ik_llama.cpp"
    if vendor_src.is_dir():
        for build_dir in ["build", "build/bin", "build/Release",
                          "build/bin/Release", "build/examples/quantize",
                          "build/examples/quantize/Release"]:
            d = vendor_src / build_dir
            for name in ["llama-quantize.exe", "llama-quantize"]:
                candidate = d / name
                if candidate.is_file():
                    return candidate

    return None


def quantize(
    input_path: str | Path,
    output_path: str | Path,
    quant_type: str = "IQ4_KT",
    imatrix_path: str | Path | None = None,
) -> Path:
    """Quantize a GGUF model using ik_llama.cpp's llama-quantize.

    Args:
        input_path: Path to the source GGUF (bf16 or f16).
        output_path: Path for the quantized output GGUF.
        quant_type: Quantization type (e.g. "IQ4_KT", "Q4_K_M").
        imatrix_path: Optional importance matrix for better quality.

    Returns:
        Path to the quantized output file.

    Raises:
        FileNotFoundError: If llama-quantize binary is not found.
        subprocess.CalledProcessError: If quantization fails.
    """
    quantize_bin = find_quantize_bin()
    if quantize_bin is None:
        raise FileNotFoundError(
            "llama-quantize not found. Ensure ik-llama-cpp-python is installed "
            "with the quantize binary, or build it from source:\n"
            "  pip install ik-llama-cpp-python  # includes llama-quantize\n"
            "  # Or build from ik_llama.cpp source:\n"
            "  cd vendor/ik_llama.cpp && mkdir build && cd build\n"
            "  cmake .. -DLLAMA_BUILD_EXAMPLES=ON && cmake --build . --target llama-quantize"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input GGUF not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(quantize_bin)]
    if imatrix_path is not None:
        imatrix_path = Path(imatrix_path)
        if not imatrix_path.is_file():
            raise FileNotFoundError(f"imatrix file not found: {imatrix_path}")
        cmd.extend(["--imatrix", str(imatrix_path)])
    cmd.extend([str(input_path), str(output_path), quant_type])

    print(f"Quantizing: {input_path.name} -> {output_path.name} ({quant_type})")
    result = subprocess.run(cmd)

    # If imatrix failed (format mismatch), retry without it
    if result.returncode != 0 and imatrix_path is not None:
        print(f"\nWarning: quantization with imatrix failed (likely format mismatch).")
        print(f"Retrying without imatrix...")
        # Clean up partial output
        if output_path.is_file():
            output_path.unlink()
        cmd_no_imat = [str(quantize_bin), str(input_path), str(output_path), quant_type]
        subprocess.run(cmd_no_imat, check=True)
    elif result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

    if not output_path.is_file():
        raise RuntimeError(f"Quantization completed but output not found: {output_path}")

    size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"Done! {output_path} ({size_gb:.2f} GB)")
    return output_path


def quantize_from_hf(
    repo_id: str,
    quant_type: str = "IQ4_KT",
    output_dir: str | Path | None = None,
) -> Path:
    """Download a bf16 GGUF + imatrix from HuggingFace and quantize.

    Expects the repo to follow bartowski's naming convention:
      - <prefix>-bf16.gguf
      - <prefix>-imatrix.gguf

    Args:
        repo_id: HuggingFace repo (e.g. "bartowski/google_gemma-4-E2B-it-GGUF").
        quant_type: Quantization type (default: "IQ4_KT").
        output_dir: Directory for downloaded and output files.

    Returns:
        Path to the quantized output file.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    # Discover bf16 and imatrix files
    files = list_repo_files(repo_id)
    bf16_files = [f for f in files if f.endswith("-bf16.gguf")]
    imatrix_files = [f for f in files if f.endswith("-imatrix.gguf")]

    if not bf16_files:
        # Fallback: try f16
        bf16_files = [f for f in files if f.endswith("-f16.gguf")]
    if not bf16_files:
        raise FileNotFoundError(
            f"No bf16/f16 source GGUF found in {repo_id}. "
            f"Available: {[f for f in files if f.endswith('.gguf')]}"
        )

    bf16_name = bf16_files[0]
    prefix = bf16_name.rsplit("-bf16.gguf", 1)[0] or bf16_name.rsplit("-f16.gguf", 1)[0]
    output_name = f"{prefix}-{quant_type}.gguf"

    if output_dir is None:
        output_dir = Path("models") / repo_id.split("/")[-1].lower().replace("-gguf", "")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name
    if output_path.is_file():
        size_gb = output_path.stat().st_size / (1024 ** 3)
        print(f"Already exists: {output_path} ({size_gb:.2f} GB)")
        return output_path

    # Download bf16
    bf16_path = output_dir / bf16_name
    if not bf16_path.is_file():
        print(f"Downloading {bf16_name} from {repo_id}...")
        hf_hub_download(repo_id=repo_id, filename=bf16_name, local_dir=str(output_dir))

    # Download imatrix (optional but recommended for IQ quants)
    imatrix_path = None
    if imatrix_files:
        imat_name = imatrix_files[0]
        imatrix_path = output_dir / imat_name
        if not imatrix_path.is_file():
            print(f"Downloading {imat_name} from {repo_id}...")
            hf_hub_download(repo_id=repo_id, filename=imat_name, local_dir=str(output_dir))

    # Quantize
    result = quantize(bf16_path, output_path, quant_type, imatrix_path)

    # Hint about cleanup
    bf16_size_gb = bf16_path.stat().st_size / (1024 ** 3)
    print(f"\nTip: delete {bf16_path.name} to save {bf16_size_gb:.1f} GB")

    return result


def main():
    parser = argparse.ArgumentParser(
        prog="ik-llama-quantize",
        description="Quantize GGUF models using ik_llama.cpp (supports IQ4_KT and other IK quants)",
    )
    sub = parser.add_subparsers(dest="command")

    # --- Direct quantize ---
    p_quant = sub.add_parser("quantize", help="Quantize a local GGUF file")
    p_quant.add_argument("input", help="Source GGUF file (bf16 or f16)")
    p_quant.add_argument("output", help="Output GGUF file path")
    p_quant.add_argument("type", nargs="?", default="IQ4_KT",
                         help="Quantization type (default: IQ4_KT)")
    p_quant.add_argument("--imatrix", help="Importance matrix file for better quality")

    # --- Download + quantize from HuggingFace ---
    p_hf = sub.add_parser("from-hf",
                          help="Download bf16 from HuggingFace and quantize")
    p_hf.add_argument("repo", help="HuggingFace repo ID (e.g. bartowski/google_gemma-4-E2B-it-GGUF)")
    p_hf.add_argument("--type", default="IQ4_KT",
                       help="Quantization type (default: IQ4_KT)")
    p_hf.add_argument("--output-dir", help="Output directory (default: models/<repo-name>)")

    # --- Check binary ---
    sub.add_parser("check", help="Check if llama-quantize binary is available")

    # Allow positional-only usage: ik-llama-quantize input output type
    args = parser.parse_args()

    if args.command is None:
        # Support positional-only usage without subcommand
        if len(sys.argv) >= 3 and not sys.argv[1].startswith("-"):
            args = argparse.Namespace(
                command="quantize",
                input=sys.argv[1],
                output=sys.argv[2],
                type=sys.argv[3] if len(sys.argv) > 3 else "IQ4_KT",
                imatrix=None,
            )
            # Check for --imatrix
            for i, a in enumerate(sys.argv):
                if a == "--imatrix" and i + 1 < len(sys.argv):
                    args.imatrix = sys.argv[i + 1]
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "check":
        b = find_quantize_bin()
        if b:
            print(f"llama-quantize found: {b}")
        else:
            print("llama-quantize not found")
            sys.exit(1)

    elif args.command == "quantize":
        quantize(args.input, args.output, args.type, args.imatrix)

    elif args.command == "from-hf":
        quantize_from_hf(args.repo, args.type, args.output_dir)


if __name__ == "__main__":
    main()
