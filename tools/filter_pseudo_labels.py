import argparse
import os

def parse_line(line):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return None
    img_path = parts[0]
    text = parts[1]
    score = None
    if len(parts) >= 3:
        try:
            score = float(parts[2])
        except ValueError:
            score = None
    return img_path, text, score


def sanitize_text(text):
    return text.replace("\t", " ").replace("\n", " ").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Filter OCR pseudo-labels by score/length.")
    parser.add_argument("--input", type=str, default="./rec_results/rec_results.txt")
    parser.add_argument("--output", type=str, default="./rec_results/pseudo_labels.txt")
    parser.add_argument("--min_score", type=float, default=0.85)
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    kept = 0
    total = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(
            args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            parsed = parse_line(line)
            if not parsed:
                continue
            img_path, text, score = parsed
            text = sanitize_text(text)
            if len(text) < args.min_len or len(text) > args.max_len:
                continue
            if score is not None and score < args.min_score:
                continue
            fout.write(f"{img_path}\t{text}\n")
            kept += 1

    print(f"Done. kept={kept}, total={total}")


if __name__ == "__main__":
    main()
