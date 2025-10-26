import argparse, json, sys, pathlib, requests

def main():
    parser = argparse.ArgumentParser(description="Local Copilot CLI")
    parser.add_argument("file", type=pathlib.Path, help="File to edit")
    parser.add_argument("-l", "--line", type=int, default=1, help="Line (1‑based) where cursor is")
    parser.add_argument("-c", "--col", type=int, default=1, help="Column (1‑based) of cursor")
    args = parser.parse_args()

    src = args.file.read_text(encoding="utf-8")
    lines = src.splitlines()
    # Compute absolute char offset
    cursor = sum(len(l)+1 for l in lines[:args.line-1]) + args.col-1

    payload = {
        "code": src,
        "cursor": cursor,
        "file_path": str(args.file)
    }
    resp = requests.post("http://127.0.0.1:8000/complete", json=payload)
    if resp.ok:
        out = resp.json()["completion"]
        print("\\n--- Completion ---\\n")
        print(out)
    else:
        print("Error:", resp.text, file=sys.stderr)

if __name__ == "__main__":
    main()