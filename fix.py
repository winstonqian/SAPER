import json

INPUT_FILE = "dataset/example_task.json"       # your broken file
OUTPUT_FILE = "dataset/example_task_fixed.json"

def load_broken_json(path):
    """
    Load a file containing multiple top-level JSON objects concatenated together.
    Splits based on balanced braces.
    """
    with open(path, "r") as f:
        text = f.read()

    objs = []
    brace_count = 0
    buffer = ""

    for char in text:
        buffer += char

        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1

        # When braces match, we have a full object
        if brace_count == 0 and buffer.strip():
            try:
                obj = json.loads(buffer)
                objs.append(obj)
                buffer = ""
            except json.JSONDecodeError:
                # Keep collecting if not a complete JSON object
                pass

    return objs


def main():
    print("Fixing JSON file…")

    objs = load_broken_json(INPUT_FILE)

    print(f"Parsed {len(objs)} valid JSON objects.")

    # Save as a proper JSON list:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(objs, f, indent=2)

    print(f"Saved fixed JSON to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()