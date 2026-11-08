"""Reproduces the demo video's real output: run, record, gate pass, gate fail."""
import sys
from teval.cli import main

if __name__ == "__main__":
    print("$ teval run examples/golden.jsonl --config examples/config.json")
    main(["run", "examples/golden.jsonl", "--config", "examples/config.json"])
    print("$ teval record examples/golden.jsonl --baseline /tmp/teval-baseline.json")
    main(["record", "examples/golden.jsonl", "--baseline", "/tmp/teval-baseline.json", "--config", "examples/config.json"])
    print("$ teval gate examples/golden.jsonl --baseline /tmp/teval-baseline.json")
    main(["gate", "examples/golden.jsonl", "--baseline", "/tmp/teval-baseline.json", "--config", "examples/config.json"])
    # /tmp/teval-regressed.jsonl: golden.jsonl with booking-clean looping hotels.search 4x (simulated bad PR)
    print("$ teval gate /tmp/teval-regressed.jsonl --baseline /tmp/teval-baseline.json  # simulated bad PR")
    sys.exit(main(["gate", "/tmp/teval-regressed.jsonl", "--baseline", "/tmp/teval-baseline.json", "--config", "examples/config.json"]))
