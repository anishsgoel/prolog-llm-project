import re
from typing import Tuple, List, Dict

from prolog.formula_parsing import split_inline_comment
from prolog.prolog_command import Fact, Rule


def parse(
    kb_text: str,
) -> Tuple[List[Fact], List[Rule], Dict[str, str]]:
    """Parse KB text into facts, rules, and predicate comments.

    Returns:
        Tuple of (facts, rules, predicate_comments)
    """
    facts = []
    rules = []
    predicate_comments: Dict[str, str] = {}

    pending_comments = []

    for line in kb_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            raise SyntaxError(f"Cannot parse line: {line}") # changed from continue

        num = int(m.group(1))
        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        if not content:
            raise SyntaxError(f"Cannot parse line: {line}") # changed from continue to see whether this can fail

        if ":-" in content:
            head, body = content.split(":-", 1)
            rule = Rule(
                num=num,
                head=head.strip(),
                body=body.strip().rstrip(".")
            )
            rules.append(rule)

            key = "{}/{}".format(rule.head_formula.pred.name, rule.head_formula.pred.arity)
            if pending_comments or inline_comment:
                combined = []
                if pending_comments:
                    combined.append(" ".join(pending_comments))
                if inline_comment:
                    combined.append(inline_comment)
                predicate_comments[key] = " ".join(combined).strip()
        else:
            fact = Fact(num=num, atom=content.rstrip("."))
            facts.append(fact)

            key = "{}/{}".format(fact.predicate_name, len(fact.args))
            if pending_comments or inline_comment:
                combined = []
                if pending_comments:
                    combined.append(" ".join(pending_comments))
                if inline_comment:
                    combined.append(inline_comment)
                predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return facts, rules, predicate_comments
