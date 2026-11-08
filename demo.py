"""Reproduces the demo video's real output: run, record, gate pass, gate fail."""
import json, sys
from teval.cli import main

def make_regressed():
    """Simulated bad-PR trajectory: booking-clean looping on flights.search."""
    bad = []
    for line in open("examples/golden.jsonl"):
        row = json.loads(line)
        if row["id"] == "booking-clean":
            steps = row["steps"]
            extra = []
            for _ in range(3):  # loop: repeat the search call + result 3 extra times
                extra += [s for s in steps if s.get("tool_call", {}).get("name") == "flights.search" or s.get("content", "").startswith("3 offers")]
            row["steps"] = steps + extra
        bad.append(json.dumps(row))
    open("/tmp/teval-regressed.jsonl", "w").write("\n".join(bad) + "\n")

if __name__ == "__main__":
    main(["record", "examples/golden.jsonl", "--baseline", "/tmp/teval-baseline.json", "--config", "examples/config.json"])
    print("$ teval gate examples/golden.jsonl --baseline /tmp/teval-baseline.json")
    main(["gate", "examples/golden.jsonl", "--baseline", "/tmp/teval-baseline.json", "--config", "examples/config.json"])
    make_regressed()
    print("$ teval gate /tmp/teval-regressed.jsonl --baseline /tmp/teval-baseline.json  # simulated bad PR")
    sys.exit(main(["gate", "/tmp/teval-regressed.jsonl", "--baseline", "/tmp/teval-baseline.json", "--config", "examples/config.json"]))
