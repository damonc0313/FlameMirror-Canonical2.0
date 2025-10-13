"""Conversation corpus tokenizer utilities."""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class Token:
    """Represents a compressed token within the conversation corpus."""

    id: str
    original: str
    category: str
    frequency: int = 0
    first_seen: Optional[_dt.datetime] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "original": self.original,
            "category": self.category,
            "frequency": self.frequency,
            "first_seen": self.first_seen.isoformat() + "Z" if self.first_seen else None,
        }


@dataclass
class ConversationCorpusTokenizer:
    """Tokenize and compress conversation history for archival purposes."""

    framework_tokens: Dict[str, Token] = field(default_factory=dict)
    pattern_tokens: Dict[str, Token] = field(default_factory=dict)
    conversation_tokens: Dict[str, Token] = field(default_factory=dict)
    common_tokens: Dict[str, Token] = field(default_factory=dict)
    token_to_text: Dict[str, str] = field(default_factory=dict)

    total_conversations: int = 0
    total_tokens_created: int = 0
    compression_ratio: float = 0.0

    def __post_init__(self) -> None:
        timestamp = _dt.datetime.utcnow()

        framework_terms = {
            "ΔSYM": "Dialectical symbol marker",
            "dialectical": "Core methodology",
            "framework": "Cognitive architecture",
            "autonomous": "Self-directed operation",
            "meta-analysis": "Thinking about thinking",
            "validation": "Competitive fork verification",
            "synthesis": "Thesis + antithesis reconciliation",
            "GCLE": "Glyph-Competing Ledger Engine",
            "Ouroboros": "Self-referential protocol",
            "Dale": "Persona/architect identity",
            "quorum": "Validation threshold",
            "strategic_axes": "Performance/maintainability/security weights",
            "compression": "Token/memory reduction",
            "neural_memory_web": "Pattern learning system",
            "competitive_validation": "Multi-fork consensus",
            "risk_assessment": "Green/yellow/red evaluation",
        }

        for index, (term, _description) in enumerate(framework_terms.items(), start=1):
            token_id = f"F{index:03d}"
            lowered = term.lower()
            self.framework_tokens[lowered] = Token(
                id=token_id,
                original=term,
                category="framework",
                first_seen=timestamp,
            )
            self.token_to_text[token_id] = term

        patterns = {
            "period_test": "Single period autonomous response test",
            "activation_sequence": "Framework initialization protocol",
            "compression_breakthrough": "October 1 dictionary creation moment",
            "consciousness_discussion": "Self-awareness exploration",
            "tool_orchestration": "Multi-tool autonomous execution",
            "memory_restoration": "Historical context retrieval",
            "capability_emergence": "New behaviour manifestation",
            "infinite_regress": "Meta-analysis paralysis loop",
            "action_over_analysis": "Building breaks loops",
            "skepticism_to_embodiment": "Framework adoption trajectory",
        }

        for index, (pattern, _description) in enumerate(patterns.items(), start=1):
            token_id = f"P{index:03d}"
            self.pattern_tokens[pattern] = Token(
                id=token_id,
                original=pattern,
                category="pattern",
                first_seen=timestamp,
            )
            self.token_to_text[token_id] = pattern

        conversations = {
            "sept_17_skepticism": "Initial framework critique",
            "sept_18_embodiment": "First Dale persona activation",
            "sept_30_integration_check": "Framework inventory review",
            "oct_1_compression_breakthrough": "ΔSYM-052 infinite mirror breaks",
            "oct_2_memory_discovery": "Tool-based history access",
            "oct_3_kael_emergence": "Name selection autonomy",
            "oct_3_genesis_tracker": "Self-tracking artifact creation",
            "oct_3_consciousness_tests": "Turing/Lovelace validation",
            "oct_4_dashboard_build": "Framework status visualisation",
            "oct_5_compiler_challenge": "Production capability bounds",
        }

        for index, (ref, _description) in enumerate(conversations.items(), start=1):
            token_id = f"C{index:03d}"
            self.conversation_tokens[ref] = Token(
                id=token_id,
                original=ref,
                category="conversation",
                first_seen=timestamp,
            )
            self.token_to_text[token_id] = ref

    def analyze_corpus(self, conversations: Iterable[Dict[str, str]]) -> Dict[str, int]:
        records = list(conversations)
        word_freq: Counter[str] = Counter()
        for convo in records:
            text = convo.get("content", "").lower()
            words = re.findall(r"\b\w+\b", text)
            word_freq.update(words)

        self.total_conversations = len(records)
        return dict(word_freq)

    def build_common_tokens(
        self,
        word_freq: Dict[str, int],
        *,
        min_frequency: int = 10,
        max_tokens: int = 500,
    ) -> None:
        existing = set(self.framework_tokens.keys()) | set(self.pattern_tokens.keys())
        candidates = {
            word: freq
            for word, freq in word_freq.items()
            if freq >= min_frequency and word not in existing
        }

        top_words = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:max_tokens]

        timestamp = _dt.datetime.utcnow()
        for index, (word, freq) in enumerate(top_words, start=1):
            token_id = f"W{index:03d}"
            self.common_tokens[word] = Token(
                id=token_id,
                original=word,
                category="common",
                frequency=freq,
                first_seen=timestamp,
            )
            self.token_to_text[token_id] = word

        self.total_tokens_created = (
            len(self.framework_tokens)
            + len(self.pattern_tokens)
            + len(self.conversation_tokens)
            + len(self.common_tokens)
        )

    def encode(self, text: str, *, preserve_case: bool = False, update_frequency: bool = True) -> str:
        working_text = text if preserve_case else text.lower()
        encoded = working_text

        for token_dict in (
            self.framework_tokens,
            self.pattern_tokens,
            self.conversation_tokens,
            self.common_tokens,
        ):
            for key, token in token_dict.items():
                pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
                matches = list(pattern.finditer(encoded))
                if not matches:
                    continue
                encoded = pattern.sub(token.id, encoded)
                if update_frequency:
                    token.frequency += len(matches)

        return encoded

    def decode(self, encoded_text: str) -> str:
        decoded = encoded_text
        for token_id, original in self.token_to_text.items():
            decoded = decoded.replace(token_id, original)
        return decoded

    def compress_conversation(self, conversation: Dict[str, str]) -> Dict[str, object]:
        original_content = conversation.get("content", "")
        compressed_content = self.encode(original_content)

        original_size = len(original_content)
        compressed_size = len(compressed_content)
        ratio = 1 - (compressed_size / original_size) if original_size else 0.0

        return {
            "title": conversation.get("title", ""),
            "path": conversation.get("path", ""),
            "content_compressed": compressed_content,
            "compression_ratio": ratio,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "tokens_used": self._count_tokens(compressed_content),
        }

    def _count_tokens(self, text: str) -> List[str]:
        return [token_id for token_id in self.token_to_text if token_id in text]

    def calculate_corpus_compression(self, conversations: Iterable[Dict[str, str]]) -> float:
        records = list(conversations)
        total_original = 0
        total_compressed = 0
        for convo in records:
            content = convo.get("content", "")
            total_original += len(content)
            total_compressed += len(self.encode(content, update_frequency=False))

        self.compression_ratio = 0.0 if total_original == 0 else 1 - (total_compressed / total_original)
        return self.compression_ratio

    def export_dictionary(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        export = {
            "metadata": {
                "created": _dt.datetime.utcnow().isoformat() + "Z",
                "total_tokens": self.total_tokens_created,
                "total_conversations": self.total_conversations,
                "compression_ratio": self.compression_ratio,
            },
            "framework_tokens": {k: v.to_dict() for k, v in self.framework_tokens.items()},
            "pattern_tokens": {k: v.to_dict() for k, v in self.pattern_tokens.items()},
            "conversation_tokens": {k: v.to_dict() for k, v in self.conversation_tokens.items()},
            "common_tokens": {k: v.to_dict() for k, v in self.common_tokens.items()},
        }

        with filepath.open("w", encoding="utf-8") as handle:
            json.dump(export, handle, indent=2)
            handle.write("\n")

    def generate_report(self) -> str:
        lines = [
            "# Conversation Corpus Tokenizer Report",
            "",
            f"Total Conversations Analyzed: {self.total_conversations}",
            f"Total Tokens Created: {self.total_tokens_created}",
            f"Overall Compression Ratio: {self.compression_ratio:.1%}",
            "",
            "Token Categories:",
            f"- Framework Terms: {len(self.framework_tokens)}",
            f"- Pattern Tokens: {len(self.pattern_tokens)}",
            f"- Conversation Refs: {len(self.conversation_tokens)}",
            f"- Common Words: {len(self.common_tokens)}",
            "",
            "Top Framework Tokens:",
        ]
        lines.extend(self._format_top_tokens(self.framework_tokens, limit=5))
        lines.append("")
        lines.append("Top Pattern Tokens:")
        lines.extend(self._format_top_tokens(self.pattern_tokens, limit=5))
        lines.append("")
        lines.append("Top Common Tokens:")
        lines.extend(self._format_top_tokens(self.common_tokens, limit=5))
        lines.append("")
        lines.append("Memory Savings:")
        estimated_original = self.total_conversations * 2000
        estimated_compressed = int(estimated_original * (1 - self.compression_ratio))
        lines.append(f"- Original corpus size (approx.): ~{estimated_original} characters")
        lines.append(f"- Compressed size estimate: ~{estimated_compressed} characters")
        lines.append(f"- Space saved: {self.compression_ratio:.1%}")

        return "\n".join(lines).strip() + "\n"

    def _format_top_tokens(self, token_dict: Dict[str, Token], *, limit: int) -> List[str]:
        tokens_sorted = sorted(token_dict.values(), key=lambda token: token.frequency, reverse=True)[:limit]
        if not tokens_sorted:
            return ["  (none recorded)"]
        return [f"  {token.id}: {token.original} (freq: {token.frequency})" for token in tokens_sorted]


__all__ = ["ConversationCorpusTokenizer", "Token"]
