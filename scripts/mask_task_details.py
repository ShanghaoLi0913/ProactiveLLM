"""
Step 2: Mask critical details to create underspecified tasks

BigCodeBench instruct_prompt 固定结构：
    [函数描述]
    Note that: [特殊说明]（可选，18.9%）
    The function should raise the exception for: [异常]（可选，26.0%）
    The function should output with:
        [返回值类型和描述]（必有，100%）
    You should write self-contained code starting with:
    ```[import + 函数签名]```（必有，100%，不 mask）

masking 策略（v29）：
- 全部基于固定结构锚点，不使用 regex，零断句残留
- output_format:     anchor "The function should output with:\n" → "You should write self-contained"
- validation_rules:  anchor "The function should raise the exception for:" → 下一个段落
- note_that:         anchor "Note that:" → 下一个段落
- edge_cases:        已移除（从 test 代码关键词匹配，不准确，用 validation_rules 替代）
"""
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.compute_task_uncertainty import compute_task_uncertainty_from_state

# 固定结构锚点
ANCHOR_OUTPUT_START = "The function should output with:\n"
ANCHOR_OUTPUT_END   = "You should write self-contained code"
ANCHOR_RAISE        = "The function should raise the exception for:"
ANCHOR_NOTE         = "Note that:"

# 段落分隔锚点（用于确定各 section 结尾）
SECTION_ANCHORS = [
    "The function should raise the exception for:",
    "The function should output with:",
    "You should write self-contained code",
    "Note that:",
]


def _find_section_end(prompt: str, start: int) -> int:
    """从 start 位置往后找到下一个 section 锚点的位置（即本 section 的结尾）。"""
    min_end = len(prompt)
    for anchor in SECTION_ANCHORS:
        idx = prompt.find(anchor, start)
        if idx != -1 and idx > start:
            min_end = min(min_end, idx)
    return min_end


def mask_prompt(instruct_prompt: str) -> Tuple[str, Dict[str, str]]:
    """
    按结构边界精确 mask output_format、validation_rules、note_that。

    Returns:
        masked_prompt:  mask 后的 prompt（零断句残留）
        masked_content: 被 mask 的各字段完整内容
            {
                "output_format":    str（必有）,
                "validation_rules": str（可选）,
                "note_that":        str（可选）,
            }
    """
    masked = instruct_prompt
    masked_content: Dict[str, str] = {}

    # ── 1. output_format（必有，100%）──────────────────────────────────────
    start_idx = masked.find(ANCHOR_OUTPUT_START)
    end_idx   = masked.find(ANCHOR_OUTPUT_END)
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        content_start = start_idx + len(ANCHOR_OUTPUT_START)
        masked_content["output_format"] = masked[content_start:end_idx].strip()
        masked = masked[:start_idx] + masked[end_idx:]

    # ── 2. validation_rules（可选，26%）────────────────────────────────────
    raise_idx = masked.find(ANCHOR_RAISE)
    if raise_idx != -1:
        content_start = raise_idx + len(ANCHOR_RAISE)
        content_end   = _find_section_end(masked, content_start)
        masked_content["validation_rules"] = masked[content_start:content_end].strip()
        masked = masked[:raise_idx] + masked[content_end:]

    # ── 3. note_that（可选，18.9%）─────────────────────────────────────────
    note_idx = masked.find(ANCHOR_NOTE)
    if note_idx != -1:
        content_start = note_idx + len(ANCHOR_NOTE)
        content_end   = _find_section_end(masked, content_start)
        note_content  = masked[content_start:content_end].strip()
        # 清理 "Notes: " 前缀（部分 task 原始是 "Note that: Notes: ..."）
        if note_content.startswith("Notes:"):
            note_content = note_content[len("Notes:"):].strip()
        masked_content["note_that"] = note_content
        masked = masked[:note_idx] + masked[content_end:]

    # 清理多余空行
    masked = re.sub(r'\n{3,}', '\n\n', masked).strip()

    return masked, masked_content


def _split_output_format(text: str) -> List[str]:
    """
    把 output_format 拆成多个 sub-items。

    OF 有稳定的 "type: description" 格式（95.7% 的任务），适合拆分。
    拆分策略：
    1. "type: description" → [Return type, 描述]
    2. dict 的 'key': desc 模式 → 每个 key 一个 item
    3. 多行描述中，独立行（大写开头+含冒号）拆分，续行合并
    4. 单行多句 → 按句子边界拆

    VR/NT 是自由文本，无稳定格式，不拆分。
    """
    text = text.strip()
    if not text:
        return []

    # ── Step 1: 找 "type: description" 分割点 ──
    # 匹配 "float:", "dict:", "Tuple[int, str]:", "List[str]:" 等
    # 类型名只包含标识符字符、方括号、圆括号、逗号、空格、点号
    type_match = re.match(r'^([\w\[\]\(\), \.]+):\s*', text)
    if not type_match or len(type_match.group(1)) > 50:
        # 无类型前缀 或 前缀太长（可能把描述误匹配了），不拆
        return [text]

    return_type = type_match.group(1).strip()
    description = text[type_match.end():].strip()

    if not description:
        return [text]

    items = [f"Return type: {return_type}"]

    # ── Step 2: 拆分描述部分 ──

    # 策略 A: dict/tuple 的 'key': desc 模式
    key_pattern = re.compile(r"['\"](\w+)['\"]:\s*(.+?)(?=\n\s*['\"]|\Z)", re.DOTALL)
    key_matches = key_pattern.findall(description)
    if len(key_matches) >= 2:
        for key_name, key_desc in key_matches:
            items.append(f"Field '{key_name}': {key_desc.strip().rstrip('.')}")
        return items

    # 策略 B: 多行描述 — 只在"独立行"处拆分
    # 独立行：大写字母开头 + 含有冒号（如 "DataFrame: ...", "Axes: ..."）
    # 或编号开头（如 "1. pandas.DataFrame: ..."）
    # 其他行视为续行，合并到前一个 item
    lines = [l.strip() for l in description.split('\n') if l.strip()]
    if len(lines) >= 2:
        merged = [lines[0]]
        for l in lines[1:]:
            is_independent = (
                re.match(r'^[\w\.]+[\w\[\]\(\), \.]*:\s', l) or  # 类型+冒号 (DataFrame: ..., plt.Axes: ..., str: ...)
                re.match(r'^\d+\.\s+', l) or                      # 编号 (1. ...)
                re.match(r'^[-•]\s+', l)                           # 列表项 (- ...)
            )
            if is_independent:
                merged.append(l)
            else:
                merged[-1] = merged[-1] + " " + l
        if len(merged) >= 2:
            items.extend(merged)
            return items

    # 策略 C: 单行多句（句号 + 大写字母开头）
    full_desc = " ".join(lines)  # 合并所有行为单行
    sentences = re.split(r'(?<=[a-z\)\]])\.\s+(?=[A-Z])', full_desc)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        # 合并过短句（<25 chars）到前一句
        merged = []
        for s in sentences:
            if merged and len(s) < 25:
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)
        if len(merged) >= 2:
            items.extend(merged)
            return items

    # 策略 D: 都不满足，描述作为整体（type + desc = 2 items）
    items.append(full_desc)
    return items


def create_disclosure_rule(masked_content: Dict[str, str]) -> Dict:
    """
    创建 disclosure_rule 数据结构。
    output_format 拆成多个 sub-items（有稳定的 type: desc 格式）。
    validation_rules / note_that 保持原样不拆（自由文本，无稳定格式）。
    disclosure_info 保留完整内容供参考。
    """
    masked_fields = {
        "output_format":    _split_output_format(masked_content.get("output_format", "")),
        "validation_rules": [masked_content["validation_rules"]] if masked_content.get("validation_rules") else [],
        "note_that":        [masked_content["note_that"]] if masked_content.get("note_that") else [],
    }

    disclosure_info = {}
    for field, content in masked_content.items():
        if content:
            disclosure_info[field] = {"specification": content}

    return {
        "masked_fields":  masked_fields,
        "disclosure_info": disclosure_info,
    }


def process_tasks(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
) -> None:
    """
    处理任务：mask output_format / validation_rules / note_that，生成 masked states。
    """
    print(f"Loading tasks from {input_path}...")
    tasks = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if limit:
        tasks = tasks[:limit]

    print(f"Loaded {len(tasks)} tasks. Masking...")

    masked_tasks = []
    stats = {"output_format": 0, "validation_rules": 0, "note_that": 0, "no_output": 0}

    for i, task in enumerate(tasks):
        original_prompt = task.get("instruct_prompt", "")
        masked_prompt, masked_content = mask_prompt(original_prompt)

        if not masked_content.get("output_format"):
            stats["no_output"] += 1
        else:
            stats["output_format"] += 1
        if masked_content.get("validation_rules"):
            stats["validation_rules"] += 1
        if masked_content.get("note_that"):
            stats["note_that"] += 1

        disclosure_rule = create_disclosure_rule(masked_content)
        task_uncertainty = compute_task_uncertainty_from_state({"query": masked_prompt})

        masked_task = {
            "id":                       task["task_id"],
            "domain":                   "coding",
            "query":                    masked_prompt,
            "dialogue_turn":            0,
            "prev_reject":              0,
            "task_uncertainty":         task_uncertainty,
            "original_instruct_prompt": original_prompt,
            "canonical_solution":       task.get("canonical_solution", ""),
            "test":                     task.get("test", ""),
            "entry_point":              task.get("entry_point", ""),
            "disclosure_rule":          disclosure_rule,
        }
        masked_tasks.append(masked_task)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tasks)} processed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for t in masked_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    total = len(masked_tasks)
    print(f"\nDone! {total} tasks saved to {output_path}")
    print(f"  output_format masked:    {stats['output_format']}/{total} ({stats['output_format']/total*100:.1f}%)")
    print(f"  validation_rules masked: {stats['validation_rules']}/{total} ({stats['validation_rules']/total*100:.1f}%)")
    print(f"  note_that masked:        {stats['note_that']}/{total} ({stats['note_that']/total*100:.1f}%)")
    if stats["no_output"]:
        print(f"  WARNING: {stats['no_output']} tasks had no output_format anchor")


def main():
    parser = argparse.ArgumentParser(description="Mask BigCodeBench tasks (v29 clean masking)")
    parser.add_argument("--input",  type=str, default="data/external/BigCodeBench/v0.1.4.jsonl")
    parser.add_argument("--output", type=str, default="data/seeds/bigcodebench_masked_states.jsonl")
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()
    process_tasks(Path(args.input), Path(args.output), limit=args.limit)


if __name__ == "__main__":
    main()
