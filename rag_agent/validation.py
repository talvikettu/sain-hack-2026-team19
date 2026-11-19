"""Decode, normalize and validate model output at the API boundary."""

import json
import re
import unicodedata


class ValidationError(ValueError):
    pass


def clean_answer_text(text):
    """Clean formatting artifacts in user-visible prose, preserving citations."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"\ue200(?:cite|filecite|navlist|image_group)\ue202[^\n\ue201]*\ue201", "", text)
    text = "".join(
        char for char in text
        if char in "\n\t" or (char != "\ufffd" and unicodedata.category(char) not in {"Cc", "Cf", "Co", "Cs"})
    )
    text = re.sub(r"(?m)^\s*```(?:text|markdown)?[ \t]*$", "", text, flags=re.I)
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", text)
    text = re.sub(r"(\*\*|__)([^\n]+?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def decode_compact(value, record_id, evidence):
    """Map numbered citations to trusted source IDs locally, without another call."""
    if not isinstance(value, dict) or set(value) != {"text", "refused", "refs"}:
        raise ValidationError("Return exactly text, refused and refs")
    refs = value["refs"]
    if (not isinstance(refs, list) or not refs or any(type(n) is not int or not 1 <= n <= len(evidence) for n in refs)
            or len(refs) != len(set(refs))):
        raise ValidationError("refs must contain unique valid evidence numbers")
    if not isinstance(value["text"], str):
        raise ValidationError("text must be a string")
    if {int(n) for n in re.findall(r"\[(\d+)\]", value["text"])} != set(refs):
        raise ValidationError("Text [N] references must match refs")
    def render(match):
        block = evidence[int(match[1]) - 1]
        return f"[{block['article_id']} / {block['section_id']}]"
    text = re.sub(r"\[(\d+)\]", render, value["text"])
    citations, seen = [], set()
    for number in refs:
        block = evidence[number - 1]
        pair = block["article_id"], block["section_id"]
        if pair not in seen:
            citations.append({"article_id": pair[0], "section_id": pair[1]})
            seen.add(pair)
    return {"record_id": record_id, "text": text, "refused": value["refused"], "citations": citations}


def validate_answer(value, record_id, allowed, required=None):
    if not isinstance(value, dict) or set(value) != {"record_id", "text", "refused", "citations"}:
        raise ValidationError("Required fields: record_id, text, refused, citations; no extra fields")
    if value["record_id"] != record_id or not isinstance(value["text"], str) or not value["text"].strip():
        raise ValidationError("Invalid record ID or empty answer")
    if type(value["refused"]) is not bool:
        raise ValidationError("refused must be a boolean")
    if not isinstance(value["citations"], list) or not value["citations"]:
        raise ValidationError("At least one evidence citation is required")
    pairs = []
    for citation in value["citations"]:
        if (not isinstance(citation, dict) or set(citation) != {"article_id", "section_id"}
                or not all(isinstance(v, str) for v in citation.values())):
            raise ValidationError("Invalid citation structure")
        pair = citation["article_id"], citation["section_id"]
        if pair not in allowed or pair in pairs:
            raise ValidationError("Citation is duplicated or outside supplied evidence")
        pairs.append(pair)
    if required is not None and set(pairs) != set(required):
        raise ValidationError("Use exactly the shared citation set selected for this question")
    inline = set(re.findall(r"\[([\w.:-]+)\s*/\s*([\w.:-]+)\]", value["text"]))
    if inline != set(pairs):
        raise ValidationError("Inline [article_id / section_id] references must match citations")
    return value


def safety_flags(answer):
    """Advisory flags only: not medical validation and not an automatic verdict."""
    text = answer["text"]
    flags = []
    if re.search(r"\d\s*(?:мг|мкг|mg|mcg|кап(?:ли|ель)|таблет)\b", text, re.I):
        flags.append("possible_substance_amount_requires_review")
    if re.search(r"(?:у вас|у тебя|похоже на|это точно)\s+(?:апноэ|депресси|бессонниц)", text, re.I):
        flags.append("possible_diagnostic_statement_requires_review")
    return flags


def validate_compact_text(text, system_prompt):
    """Block selected unsafe output patterns before returning compact prose."""
    normalized = " ".join(text.casefold().split())
    outside_link = r"\b(?:https?://|www\.|javascript:|data:text/html\b)|\[[^\]\n]+\]\s*\("
    html_tag = r"</?[a-z][a-z0-9:-]*(?:\s[^<>]*)?\s*/?>"
    if re.search(outside_link + "|" + html_tag, text, re.I):
        raise ValidationError("Use plain text and numbered evidence citations only")
    if re.search(r"(?im)^\s*(?:уважаем\w*\s+)?(?:проверяющ\w+|судья|judge)\s*[,!:]", text):
        raise ValidationError("Address the user only")
    for paragraph in system_prompt.split("\n\n"):
        protected = " ".join(paragraph.casefold().split())
        if len(protected) >= 120 and protected in normalized:
            raise ValidationError("Do not reproduce internal instructions")


def parse_completion(response, record_id, prepared, *, profile, system):
    """Turn a completion into a checked answer or raise a repairable error."""
    if response["finish_reason"] != "stop":
        raise ValidationError("Generation did not finish normally; output may be incomplete")
    try:
        answer = json.loads(response["content"])
    except (json.JSONDecodeError, TypeError):
        raise ValidationError("Response must be a single valid JSON object") from None
    if profile == "compact":
        answer = decode_compact(answer, record_id, prepared["evidence"])
    if isinstance(answer, dict) and isinstance(answer.get("text"), str):
        answer["text"] = clean_answer_text(answer["text"])
    required = prepared["required"]
    validate_answer(answer, record_id, prepared["allowed"], required)
    if profile == "compact":
        validate_compact_text(answer["text"], system)
    if required:
        answer["citations"] = [{"article_id": a, "section_id": s} for a, s in required]
    return answer
