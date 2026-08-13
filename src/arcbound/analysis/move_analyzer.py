"""Move quality analysis for replayed games."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MoveAnalysis:
    """Result of analyzing a single move."""

    turn: int
    phase: str
    player: str
    decision_type: str
    description: str
    action_taken: str
    score: int  # 0-100
    rating: str  # "Blunder", "Mistake", "Inaccuracy", "Good", "Best"
    confidence: Optional[float] = None
    value_estimate: Optional[float] = None
    timeout: bool = False


class MoveAnalyzer:
    """Analyze move quality from replay decisions.

    Scoring:
    - Score 95-100: Best move (matches model's top choice)
    - Score 80-95: Good move (near-optimal)
    - Score 60-80: Inaccuracy (slightly suboptimal)
    - Score 30-60: Mistake (below average)
    - Score 0-30: Blunder (significantly worse)
    - Timeout/fallback: Score 0
    """

    @staticmethod
    def rate_score(score: int) -> str:
        """Convert numeric score to rating label."""
        if score >= 95:
            return "Best"
        elif score >= 80:
            return "Good"
        elif score >= 60:
            return "Inaccuracy"
        elif score >= 30:
            return "Mistake"
        else:
            return "Blunder"

    @staticmethod
    def analyze_decisions(decisions: List[Dict]) -> List[MoveAnalysis]:
        """Analyze a list of decision dicts from a replay.

        For now, scores based on confidence and timeout status.
        When a real model exists, this will compare chosen action to model's policy.
        """
        analyses = []
        for dec in decisions:
            turn = dec.get("turn", 0)
            phase = dec.get("phase", "")
            player = dec.get("player", "")
            decision_type = dec.get("decision_type", "")
            description = dec.get("description", "")
            action = dec.get("action_taken", "")
            confidence = dec.get("model_confidence")
            value_est = dec.get("model_value_estimate")
            timeout = dec.get("timeout", False)

            if timeout:
                score = 0
            elif confidence is not None:
                # Use confidence as proxy for move quality
                score = int(confidence * 100)
            else:
                # Unknown quality - assume neutral
                score = 50

            analyses.append(MoveAnalysis(
                turn=turn,
                phase=phase,
                player=player,
                decision_type=decision_type,
                description=description,
                action_taken=action,
                score=score,
                rating=MoveAnalyzer.rate_score(score),
                confidence=confidence,
                value_estimate=value_est,
                timeout=timeout,
            ))
        return analyses

    @staticmethod
    def summarize(analyses: List[MoveAnalysis]) -> Dict[str, Any]:
        """Produce a summary of move analyses."""
        if not analyses:
            return {
                "total": 0,
                "best": 0,
                "good": 0,
                "inaccuracy": 0,
                "mistake": 0,
                "blunder": 0,
                "average_score": 0.0,
            }

        total = len(analyses)
        counts: Dict[str, int] = {"Best": 0, "Good": 0, "Inaccuracy": 0, "Mistake": 0, "Blunder": 0}
        for a in analyses:
            counts[a.rating] = counts.get(a.rating, 0) + 1

        avg = sum(a.score for a in analyses) / total

        return {
            "total": total,
            "best": counts["Best"],
            "good": counts["Good"],
            "inaccuracy": counts["Inaccuracy"],
            "mistake": counts["Mistake"],
            "blunder": counts["Blunder"],
            "average_score": round(avg, 1),
        }
