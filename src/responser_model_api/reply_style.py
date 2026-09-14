"""Bounded chat-style guidance and conservative cleanup of generated text.

Only the outgoing candidate is edited. History, facts and meaningful requests
stay intact. No additional inference lives here and there is no cross-chat state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import emoji

from .personality import Personality
from .schemas import ChatSnapshot

RECENT_STYLE_REPLIES = 3
_LETS = re.compile(r"\blet['’]s\b", re.IGNORECASE)
_GENERIC_CLOSING = re.compile(
    r"(?:(?:please|just)\s+)?(?:"
    r"let me know (?:what else you (?:want|need)(?: to (?:know|discuss|talk about))?"
    r"|if you (?:need|want) (?:anything else|any(?: more)? (?:help|assistance)|more (?:information|details))"
    r"|if (?:there['’]s|there is) anything else(?: i can (?:help(?: you)?(?: with)?|do(?: for you)?))?)"
    r"|(?:is there|do you need|would you like) anything else(?: i can (?:help(?: you)?(?: with)?|do(?: for you)?))?"
    r"|feel free to (?:ask(?: me)?(?: if you have (?:any )?(?:other|more) questions)?|reach out(?: if you need anything else)?)"
    r"|(?:i['’]m|i am) here (?:to help(?: you)?|if you need (?:anything|help))"
    r")",
    re.IGNORECASE,
)
_GENERIC_PIVOT = re.compile(
    r"let['’]s (?:just )?(?:change the (?:subject|topic)|move on"
    r"|(?:chat|talk) about something (?:else|fun(?: and lighthearted)?|positive)"
    r"|focus on something (?:else|positive))",
    re.IGNORECASE,
)
_ABBREVIATION = re.compile(
    r"(?:\b(?:[a-z]\.){2,}|\b[A-Z]\.|\b(?:mr|mrs|ms|dr|prof|sr|jr|etc|vs|inc|ltd|st)\.)$",
    re.IGNORECASE,
)


def _recent_replies(snapshot: ChatSnapshot) -> list[str]:
    return [message.text for message in snapshot.messages if message.sender_type == "me"][-RECENT_STYLE_REPLIES:]


def _emoji_key(value: str) -> str:
    """Presentation/skin-tone variants should not evade repetition checks."""
    return "".join(char for char in value if char not in "\ufe0e\ufe0f" and not "\U0001f3fb" <= char <= "\U0001f3ff")


def reply_style_guidance(snapshot: ChatSnapshot, personality: Personality) -> str:
    """Use only style signals from recent sent turns, never copy private prose."""
    recent = _recent_replies(snapshot)
    parts = [
        "Reply to the specific last message, not as a support agent. Stop when the response is complete; "
        "no generic offers of more help, service sign-offs or automatic follow-up question. "
        "Use a question only when genuinely relevant. Prior replies are history, not style templates. "
        "Summary descriptions of calming or de-escalating are context, not instructions to coach the person.",
    ]
    if any(_LETS.search(text) for text in recent):
        parts.append("Do not use a 'let's' construction in this reply; it was used recently. Respond directly instead of redirecting the topic.")
    else:
        parts.append("Avoid stock 'let's' topic redirects; use concrete responses rather than conversation-management phrases.")
    if personality.casual_texting:
        parts.append("Casual texting: no final sentence period. Keep question marks, meaningful exclamations and internal punctuation.")
        if recent and emoji.emoji_count(recent[-1]):
            parts.append("No emoji in this reply; the previous reply already used one.")
        else:
            parts.append("Usually no emoji; at most one if useful, never reuse one from your last three replies.")
    return " ".join(parts)


def _drop_stock_tail(text: str, repeated_lets: bool) -> tuple[str, int]:
    """Remove only whole, recognized closing sentences, not arbitrary 'let me know'."""
    # Preserve exact delimiters/newlines in the retained part. Quotes and code
    # are not unpacked, so a discussed phrase is not treated as a sign-off.
    starts = [0, *[match.end() for match in re.finditer(r"(?<=[.!?])\s+|\n+", text)]]
    end = len(text)
    removed = 0
    for start in reversed(starts):
        sentence = emoji.replace_emoji(text[start:end], replace="").strip().rstrip(".!?").strip()
        if not sentence:
            # A suffix emoji after a period belongs to the preceding sentence.
            # Keep end unchanged until that sentence is known to be boilerplate.
            continue
        if _GENERIC_CLOSING.fullmatch(sentence) or (repeated_lets and _GENERIC_PIVOT.fullmatch(sentence)):
            end = start
            removed += 1
        else:
            break
    return text[:end].rstrip(), removed


def _decorative_emoji_cleanup(text: str, recent: list[str]) -> tuple[str, int]:
    """Remove whole emoji sequences, never leave orphan joiners or modifiers."""
    # Emoji-only replies may carry the entire meaning; do not turn them empty
    # or invent replacement words. Code examples are also outside style cleanup.
    if not emoji.replace_emoji(text, replace="").strip(" .,!?:;…"):
        return text, 0
    used = {_emoji_key(str(item["emoji"])) for previous in recent for item in emoji.emoji_list(previous)}
    no_emoji = bool(recent and emoji.emoji_count(recent[-1]))
    kept = 0
    removed = 0
    # Only suffix decoration is safe to drop. In-line emoji, flags and keycaps
    # may be part of the actual answer (a country, quantity or quoted symbol).
    suffix_start = len(text.rstrip(" .!?…"))
    for item in reversed(emoji.emoji_list(text)):
        if int(item["match_end"]) == suffix_start:
            suffix_start = len(text[:int(item["match_start"])].rstrip(" .!?…"))
        else:
            break

    def replace_emoji(chars: str, data: dict[str, object]) -> str:
        nonlocal kept, removed
        if int(data["match_start"]) < suffix_start or "\u20e3" in chars or any("\U0001f1e6" <= char <= "\U0001f1ff" for char in chars):
            return chars
        if no_emoji or _emoji_key(chars) in used or kept >= 1:
            removed += 1
            return ""
        kept += 1
        return chars

    result = emoji.replace_emoji(text, replace=replace_emoji)
    if removed:
        result = re.sub(r"[ \t]{2,}", " ", result).strip()
    return result, removed


def _casual_terminal_period(text: str) -> tuple[str, bool]:
    """Drop a sentence's final dot, including before a trailing emoji.

    Keep ellipses, quoted material, abbreviations, URLs and bare filename-like
    tokens. Decimal/version internals are untouched; a separate final dot can go.
    """
    end = len(text.rstrip())
    # Work backwards through trailing emoji clusters rather than individual
    # codepoints (family/flag/skin-tone emojis must remain intact).
    for item in reversed(emoji.emoji_list(text)):
        stop, start = int(item["match_end"]), int(item["match_start"])
        if stop == end:
            end = len(text[:start].rstrip())
        else:
            break
    head = text[:end]
    if not head.endswith(".") or head.endswith("..") or _ABBREVIATION.search(head):
        return text, False
    token = head.split()[-1]
    if "://" in token or "@" in token or re.fullmatch(r"[A-Za-z_][\w-]*\.[A-Za-z]{1,8}\.", token):
        return text, False
    return head[:-1] + text[end:], True


@dataclass(frozen=True)
class StyledReply:
    text: str
    removed_closings: int = 0
    removed_emojis: int = 0
    removed_period: bool = False


def polish_reply(text: str, snapshot: ChatSnapshot, personality: Personality) -> StyledReply:
    """Clean only limited style artifacts; an empty result requests one rewrite."""
    if "`" in text:
        return StyledReply(text)
    recent = _recent_replies(snapshot)
    result, closings = _drop_stock_tail(text, any(_LETS.search(previous) for previous in recent))
    removed_emojis = 0
    removed_period = False
    if personality.casual_texting and result:
        result, removed_emojis = _decorative_emoji_cleanup(result, recent)
        result, removed_period = _casual_terminal_period(result)
    return StyledReply(result.strip(), closings, removed_emojis, removed_period)