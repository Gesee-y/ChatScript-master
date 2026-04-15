from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "DATA"


def parse_raw_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for raw_line in parse_raw_lines(path):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        triples.append((parts[0], parts[1], parts[2]))
    return triples


def difficulty_label(answer_count: int, relation_support: int, head_degree: int) -> str:
    if answer_count >= 4 or (answer_count >= 2 and relation_support >= 6):
        return "hard"
    if answer_count >= 2 or head_degree >= 4:
        return "medium"
    return "easy"


def choose_eval_pairs(
    triples: list[tuple[str, str, str]],
) -> tuple[list[dict[str, object]], set[tuple[str, str, str]], Counter[str]]:
    pair_to_tails: dict[tuple[str, str], list[str]] = defaultdict(list)
    relation_frequency: Counter[str] = Counter()
    head_relation_set: defaultdict[str, set[str]] = defaultdict(set)
    head_frequency: Counter[str] = Counter()

    for head, relation, tail in triples:
        pair_to_tails[(head, relation)].append(tail)
        relation_frequency[relation] += 1
        head_relation_set[head].add(relation)
        head_frequency[head] += 1

    eval_rows: list[dict[str, object]] = []
    withheld_triples: set[tuple[str, str, str]] = set()

    ranked_pairs = sorted(
        pair_to_tails.items(),
        key=lambda item: (
            -len(item[1]),
            -relation_frequency[item[0][1]],
            -len(head_relation_set[item[0][0]]),
            item[0][0],
            item[0][1],
        ),
    )

    for head_relation, tails in ranked_pairs:
        head, relation = head_relation
        unique_tails = sorted(set(tails))
        head_degree = len(head_relation_set[head])
        relation_support = relation_frequency[relation]

        if len(unique_tails) >= 2:
            withheld = unique_tails[1:]
            kept = unique_tails[:1]
        elif head_frequency[head] >= 2:
            withheld = unique_tails
            kept = []
        else:
            continue

        for tail in withheld:
            withheld_triples.add((head, relation, tail))

        eval_rows.append(
            {
                "head": head,
                "relation": relation,
                "valid_tails": withheld,
                "num_valid_tails": len(withheld),
                "kept_train_tails": kept,
                "difficulty": difficulty_label(len(withheld), relation_support, head_degree),
                "relation_support": relation_support,
                "head_degree": head_degree,
            }
        )

    eval_rows.sort(
        key=lambda row: (
            {"easy": 0, "medium": 1, "hard": 2}[str(row["difficulty"])],
            int(row["num_valid_tails"]),
            int(row["relation_support"]),
            int(row["head_degree"]),
            str(row["head"]),
            str(row["relation"]),
        )
    )
    return eval_rows, withheld_triples, relation_frequency


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pruned_train(
    input_path: Path,
    output_path: Path,
    withheld_triples: set[tuple[str, str, str]],
) -> tuple[int, int]:
    original_lines = parse_raw_lines(input_path)
    kept_lines: list[str] = []
    removed_count = 0
    triple_count = 0

    for raw_line in original_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            kept_lines.append(raw_line)
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            kept_lines.append(raw_line)
            continue
        triple = (parts[0], parts[1], parts[2])
        triple_count += 1
        if triple in withheld_triples:
            removed_count += 1
            continue
        kept_lines.append(raw_line)

    output_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return triple_count, removed_count


def main() -> None:
    for input_path in sorted(DATA_DIR.glob("train_instance_*.txt")):
        triples = parse_triples(input_path)
        eval_rows, withheld_triples, _ = choose_eval_pairs(triples)

        stem = input_path.stem
        pruned_train_path = input_path.with_name(f"{stem}_train_pruned.txt")
        questions_path = input_path.with_name(f"{stem}_questions.csv")
        answers_path = input_path.with_name(f"{stem}_answers.csv")

        total_triples, removed_triples = write_pruned_train(input_path, pruned_train_path, withheld_triples)

        question_rows: list[dict[str, str]] = []
        answer_rows: list[dict[str, str]] = []

        for index, row in enumerate(eval_rows, start=1):
            question_id = f"Q{index:04d}"
            question_rows.append(
                {
                    "question_id": question_id,
                    "head": str(row["head"]),
                    "relation": str(row["relation"]),
                }
            )
            answer_rows.append(
                {
                    "question_id": question_id,
                    "head": str(row["head"]),
                    "relation": str(row["relation"]),
                    "valid_tails": "|".join(str(tail) for tail in row["valid_tails"]),
                    "num_valid_tails": str(row["num_valid_tails"]),
                    "difficulty": str(row["difficulty"]),
                    "kept_train_tails": "|".join(str(tail) for tail in row["kept_train_tails"]),
                }
            )

        write_csv(questions_path, ["question_id", "head", "relation"], question_rows)
        write_csv(
            answers_path,
            ["question_id", "head", "relation", "valid_tails", "num_valid_tails", "difficulty", "kept_train_tails"],
            answer_rows,
        )

        print(
            f"{input_path.name}: questions={len(question_rows)} "
            f"removed_triples={removed_triples} kept_triples={total_triples - removed_triples}"
        )


if __name__ == "__main__":
    main()
