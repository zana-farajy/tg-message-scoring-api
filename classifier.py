import os
import re
import yaml
from typing import Dict, Any, List, Tuple
from normalizer import normalize_text
from schemas import ClassifyResponse

class MessageClassifier:
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.rules = {}
        self.load_rules()

    def load_rules(self) -> None:
        """Loads configuration rules from the rules.yaml file."""
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(f"Rules configuration file not found at: {self.rules_path}")
        with open(self.rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f)

    def is_only_link(self, text: str) -> bool:
        """Checks if the message consists only of links and spaces/punctuation."""
        links = re.findall(r'https?://[^\s]+|t\.me/[^\s]+|www\.[^\s]+', text)
        if not links:
            return False
        cleaned = text
        for link in links:
            cleaned = cleaned.replace(link, "")
        # Remove common punctuation and spaces
        cleaned = re.sub(r'[\s.,!?;:()\[\]{}_#@\-+*/«»]*', '', cleaned)
        return len(cleaned) == 0

    def is_only_emoji(self, text: str) -> bool:
        """Checks if the message consists only of emojis and spaces/punctuation."""
        cleaned = re.sub(r'[\s.,!?;:()\[\]{}_#@\-+*/«»]*', '', text)
        if not cleaned:
            return False
        # Broad emoji and symbol unicode ranges
        emoji_pattern = re.compile(
            r'[\u2600-\u27BF]|'
            r'[\u2300-\u23FF]|'
            r'[\u2b50\u2b06\u2192\u2190]|'
            r'[\U0001F000-\U0001F9FF]|'
            r'[\U0001FA00-\U0001FAFF]'
        )
        remaining = emoji_pattern.sub('', cleaned)
        return len(remaining) == 0

    def is_only_digits_or_code(self, text: str) -> bool:
        """Checks if the message consists only of digits, mathematical operators, and code formatting characters."""
        cleaned = re.sub(r'[\s0-9۰-۹٠-٩\-+*/=()__{};[\]<>\'"`#:$%&|\\~^?.,!?;:«»]*', '', text)
        return len(cleaned) == 0

    def get_link_hashtag_mention_ratio(self, text: str) -> float:
        """Calculates the ratio of links, hashtags, and mentions lengths relative to the total text length."""
        if not text:
            return 0.0
        links = re.findall(r'https?://[^\s]+|t\.me/[^\s]+|www\.[^\s]+', text)
        hashtags = re.findall(r'#[^\s]+', text)
        mentions = re.findall(r'@[^\s]+', text)
        
        total_special_len = sum(len(x) for x in links + hashtags + mentions)
        return total_special_len / len(text)

    def _count_keyword_matches(self, text: str, keyword: str) -> int:
        """
        Performs boundary-aware keyword matching for Persian and English.
        For Persian, handles common grammatical inflections gracefully.
        """
        # Character set of all alphanumeric characters in Persian and English + ZWNJ
        word_chars = r'a-zA-Z0-9\u0600-\u06FF\u0750-\u077F\ufb50-\ufdff\ufe70-\ufeff\u200c'
        
        # Persian common word suffixes
        persian_suffixes = r'(?:\u200c?(?:م|ی|یم|ید|ند|ه|ها|ان|تر|ترین|های|هایی|ام|ات|اش|مان|تان|شان))?'
        
        pattern = (
            rf'(?<![{word_chars}])'
            rf'{re.escape(keyword.lower())}'
            rf'{persian_suffixes}'
            rf'(?![{word_chars}])'
        )
        
        matches = re.findall(pattern, text)
        return len(matches)

    def classify(self, raw_text: str, chat_title: str = None, sender_name: str = None) -> ClassifyResponse:
        """
        Classifies a raw Telegram message and returns a detailed evaluation.
        """
        # 1. Fast Reject - Basic Checks on raw/un-normalized text first
        if not raw_text or not raw_text.strip():
            return ClassifyResponse(
                valuable=False,
                category="irrelevant",
                score=0.0,
                confidence=0.2,
                reasons=["Message is empty"],
                matched_keywords=[],
                negative_reasons=["Empty message rejected"]
            )

        trimmed_text = raw_text.strip()
        fast_reject_rules = self.rules.get("fast_reject", {})
        min_char_length = fast_reject_rules.get("min_char_length", 20)

        if len(trimmed_text) < min_char_length:
            return ClassifyResponse(
                valuable=False,
                category="irrelevant",
                score=0.0,
                confidence=0.2,
                reasons=[f"Message length is less than {min_char_length} characters ({len(trimmed_text)})"],
                matched_keywords=[],
                negative_reasons=["Message too short"]
            )

        if self.is_only_link(trimmed_text):
            return ClassifyResponse(
                valuable=False,
                category="irrelevant",
                score=0.0,
                confidence=0.2,
                reasons=["Message consists only of links"],
                matched_keywords=[],
                negative_reasons=["Link-only message rejected"]
            )

        if self.is_only_emoji(trimmed_text):
            return ClassifyResponse(
                valuable=False,
                category="irrelevant",
                score=0.0,
                confidence=0.2,
                reasons=["Message consists only of emojis"],
                matched_keywords=[],
                negative_reasons=["Emoji-only message rejected"]
            )

        if self.is_only_digits_or_code(trimmed_text):
            return ClassifyResponse(
                valuable=False,
                category="irrelevant",
                score=0.0,
                confidence=0.2,
                reasons=["Message consists only of numbers or code-like symbols"],
                matched_keywords=[],
                negative_reasons=["Digits/code-only message rejected"]
            )

        ratio = self.get_link_hashtag_mention_ratio(trimmed_text)
        max_link_ratio = fast_reject_rules.get("max_link_ratio", 0.60)
        if ratio > max_link_ratio:
            return ClassifyResponse(
                valuable=False,
                category="irrelevant",
                score=0.0,
                confidence=0.2,
                reasons=[f"Over {max_link_ratio * 100}% of the message consists of links/hashtags/mentions ({ratio:.1%})"],
                matched_keywords=[],
                negative_reasons=["Excessive links/hashtags/mentions"]
            )

        # 2. Normalization for keyword matching
        normalized_text = normalize_text(trimmed_text)

        # 3. Fast Reject - Spam Keywords in Normalized Text
        spam_keywords = fast_reject_rules.get("spam_keywords", [])
        matched_spam = []
        for word in spam_keywords:
            if self._count_keyword_matches(normalized_text, word) > 0:
                matched_spam.append(word)

        if matched_spam:
            return ClassifyResponse(
                valuable=False,
                category="irrelevant",
                score=0.0,
                confidence=0.2,
                reasons=[f"Message contains highly blacklisted spam keywords: {', '.join(matched_spam)}"],
                matched_keywords=[],
                negative_reasons=[f"Matched severe spam keywords: {matched_spam}"]
            )

        # 4. Positive Scoring
        category_scores = {}
        category_matched_keywords = {}
        category_matched_modifiers = {}
        all_matched_keywords_set = set()

        for category, config in self.rules.get("positive_categories", {}).items():
            core_keywords = config.get("keywords", [])
            keyword_score_val = config.get("keyword_score", 3.0)
            modifiers = config.get("modifiers", [])
            modifier_score_val = config.get("modifier_score", 2.0)

            # Find matching core keywords
            matched_cores = []
            for kw in core_keywords:
                if self._count_keyword_matches(normalized_text, kw) > 0:
                    matched_cores.append(kw)
                    all_matched_keywords_set.add(kw)

            if matched_cores:
                base_score = keyword_score_val * len(matched_cores)

                # Find matching modifiers
                matched_mods = []
                for mod in modifiers:
                    if self._count_keyword_matches(normalized_text, mod) > 0:
                        matched_mods.append(mod)
                        # Only count as matched keyword if not already matched
                        all_matched_keywords_set.add(mod)

                modifier_score_total = modifier_score_val * len(matched_mods)
                category_scores[category] = base_score + modifier_score_total
                category_matched_keywords[category] = matched_cores
                category_matched_modifiers[category] = matched_mods
            else:
                category_scores[category] = 0.0
                category_matched_keywords[category] = []
                category_matched_modifiers[category] = []

        # 5. Negative Scoring
        negative_score = 0.0
        matched_negative_keywords = []
        negative_reasons = []

        neg_scoring_config = self.rules.get("negative_scoring", {})
        default_penalty = neg_scoring_config.get("default_penalty", 2.0)

        for neg_cat, neg_config in neg_scoring_config.get("categories", {}).items():
            penalty = neg_config.get("penalty", default_penalty)
            keywords = neg_config.get("keywords", [])

            matched_in_cat = []
            for kw in keywords:
                if self._count_keyword_matches(normalized_text, kw) > 0:
                    matched_in_cat.append(kw)

            if matched_in_cat:
                category_penalty = penalty * len(matched_in_cat)
                negative_score += category_penalty
                matched_negative_keywords.extend(matched_in_cat)
                negative_reasons.append(f"Matched negative category '{neg_cat}': {', '.join(matched_in_cat)} (-{category_penalty})")

        # 6. Category Selection & Tie Breaking
        winning_category = "irrelevant"
        max_pos_score = 0.0

        for category, score in category_scores.items():
            if score > max_pos_score:
                max_pos_score = score
                winning_category = category
            elif score > 0 and score == max_pos_score:
                # Tie-breaking by stronger evidence
                curr_evidence = len(category_matched_keywords[category]) + len(category_matched_modifiers[category])
                prev_evidence = len(category_matched_keywords[winning_category]) + len(category_matched_modifiers[winning_category])
                if curr_evidence > prev_evidence:
                    winning_category = category

        positive_score = max_pos_score
        final_score = positive_score - negative_score

        # 7. Decision Making
        valuable = False
        reasons = []

        thresholds = self.rules.get("thresholds", {})
        min_score_valuable = thresholds.get("min_score_valuable", 8)
        min_score_maybe = thresholds.get("min_score_maybe_valuable", 5)
        max_score_maybe = thresholds.get("max_score_maybe_valuable", 7)
        min_cat_matches = thresholds.get("min_category_matches_for_maybe", 2)
        max_neg_allowed = thresholds.get("max_negative_score_allowed", 4)

        if winning_category == "irrelevant" or positive_score == 0:
            valuable = False
            reasons.append("No relevant business or project indicators matched.")
        elif negative_score > max_neg_allowed:
            valuable = False
            reasons.append(f"Rejected due to high negative score ({negative_score} > allowed {max_neg_allowed}).")
        else:
            if final_score < min_score_maybe:
                valuable = False
                reasons.append(f"Final score {final_score} is below the minimum threshold of {min_score_maybe}.")
            elif min_score_maybe <= final_score <= max_score_maybe:
                num_matches = len(category_matched_keywords[winning_category]) + len(category_matched_modifiers[winning_category])
                if num_matches >= min_cat_matches:
                    valuable = True
                    reasons.append(f"Moderate score ({final_score}) with strong indicators ({num_matches} matches).")
                else:
                    valuable = False
                    reasons.append(f"Moderate score ({final_score}) but insufficient indicators ({num_matches} match(es), requires >= {min_cat_matches}).")
            elif final_score >= min_score_valuable:
                valuable = True
                reasons.append(f"High final score ({final_score}) indicating highly valuable content.")

        # If it's valuable, build positive highlights
        if winning_category != "irrelevant":
            pos_highlights = []
            if category_matched_keywords[winning_category]:
                pos_highlights.append(f"Matched core '{winning_category}' keyword(s): {', '.join(category_matched_keywords[winning_category])}")
            if category_matched_modifiers[winning_category]:
                pos_highlights.append(f"Matched '{winning_category}' modifier(s): {', '.join(category_matched_modifiers[winning_category])}")
            reasons = pos_highlights + reasons

        # 8. Confidence calculation
        mapping_score = max(0.0, final_score)
        if mapping_score < 5:
            confidence = 0.2 + (mapping_score / 5.0) * 0.29
        elif 5 <= mapping_score <= 7:
            confidence = 0.5 + ((mapping_score - 5) / 2.0) * 0.24
        elif 8 <= mapping_score <= 11:
            confidence = 0.75 + ((mapping_score - 8) / 3.0) * 0.14
        else:
            confidence = min(0.99, 0.9 + ((mapping_score - 12) / 8.0) * 0.09)

        # Reduce confidence if there are negative indicators
        if negative_score > 0:
            confidence = max(0.1, confidence - (0.05 * negative_score))

        confidence = round(confidence, 2)

        return ClassifyResponse(
            valuable=valuable,
            category=winning_category,
            score=final_score,
            confidence=confidence,
            reasons=reasons,
            matched_keywords=sorted(list(all_matched_keywords_set)),
            negative_reasons=negative_reasons
        )
