#!/usr/bin/env python3
"""Check production dependency direction; development-only oracles may cross it."""
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "foundation": {
        "engine-core",
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn",
        "ribn-batch",
        "ribn-text",
        "ribn-cli",
    },
    "batch": {
        "engine-core",
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn",
        "ribn-text",
        "ribn-cli",
    },
    "runtime": {"engine-core", "engine-gguf", "engine-nvidia", "engine-qwen", "ribn-text", "ribn-cli"},
    "core": {"engine-gguf", "engine-nvidia", "engine-qwen", "ribn", "ribn-text", "ribn-cli"},
    "gguf": {"engine-nvidia", "engine-qwen", "ribn", "ribn-text", "ribn-cli"},
    "qwen": {"ribn-text", "ribn-cli"},
}


def dependency_names(manifest: dict) -> set[str]:
    sections = [manifest, *manifest.get("target", {}).values()]
    return {
        spec.get("package", name) if isinstance(spec, dict) else name
        for section in sections
        for kind in ("dependencies", "build-dependencies")
        for name, spec in section.get(kind, {}).items()
    }


def main() -> int:
    errors = []
    for crate, forbidden in FORBIDDEN.items():
        path = ROOT / "crates" / crate / "Cargo.toml"
        manifest = tomllib.loads(path.read_text())
        invalid = dependency_names(manifest) & forbidden
        if invalid:
            errors.append(f"{path.relative_to(ROOT)}: reversed dependency: {sorted(invalid)}")
    for obsolete in ("crates/core/src/nvidia.rs", "crates/gguf/src/qwen.rs"):
        if (ROOT / obsolete).exists():
            errors.append(f"{obsolete}: implementation is back in the wrong owner")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Production dependency boundaries passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
