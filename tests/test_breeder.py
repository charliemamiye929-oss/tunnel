"""The Claude-driven breeding path, exercised with a stubbed client."""
import random

from evotrader.breeder import HybridBreeder, LLMBreeder, MutationBreeder
from evotrader.fitness import Evaluation, Metrics
from evotrader.genome import Genome, is_valid
from evotrader.journal import Journal, Trade
from evotrader.llm import LLMError, Response
from evotrader.prompts import GENOME_JSON_SCHEMA, build_breeding_prompt


class FakeClaude:
    """Stands in for the API: records prompts, returns a canned payload."""

    def __init__(self, payload, *, fail=False):
        self.payload = payload
        self.fail = fail
        self.prompts = []
        self.available = True
        self.unavailable_reason = ""

    def budget_left(self):
        return float("inf")

    def complete(self, prompt, *, system="", schema=None, **kwargs):
        self.prompts.append((prompt, system, schema))
        if self.fail:
            raise LLMError("rate limited")
        return Response(text="", data=self.payload, model="fake", cost_usd=0.01)


def _payload(n=3, include_broken=False):
    genomes = [{
        "name": f"Offspring {i}",
        "thesis": "buy pullbacks in an uptrend",
        "rationale": "the elites all made money buying dips above the 200-day",
        "parents": ["abc123"],
        "entry_rules": [{"when": f"rsi14 < {30 + i} and close > sma200", "weight": 0.25}],
        "exit_rules": [{"when": "rsi14 > 62 or position_return < -0.07"}],
        "risk": {"max_position_pct": 0.3, "max_positions": 4, "max_gross_exposure": 1.0,
                 "stop_loss_pct": 0.08, "take_profit_pct": 0.0, "trailing_stop_pct": 0.0,
                 "max_hold_bars": 0, "min_hold_bars": 0, "cooldown_bars": 2},
    } for i in range(n)]
    if include_broken:
        genomes.append({"name": "Broken", "thesis": "", "rationale": "", "parents": [],
                        "entry_rules": [{"when": "secret_alpha > 1", "weight": 0.5}],
                        "exit_rules": [{"when": "rsi14 > 70"}], "risk": {}})
        genomes.append({"name": "No rules", "entry_rules": [], "exit_rules": []})
    return {"analysis": "The winners bought fear inside uptrends.",
            "lessons": ["oversold entries need a trend filter"], "genomes": genomes}


def _elite():
    genome = Genome.from_dict({
        "name": "Dip Buyer", "id": "abc123",
        "entry_rules": [{"when": "rsi14 < 32 and close > sma200", "weight": 0.3}],
        "exit_rules": ["rsi14 > 60"]})
    journal = Journal()
    journal.trades.append(Trade("SPY", "2020-03-20", "2020-04-14", 100.0, 118.0, 10.0,
                                180.0, 0.18, 17, "entry rule: rsi14 < 32", "exit rule: rsi14 > 60"))
    journal.record_hit("rsi14 < 32 and close > sma200")
    ev = Evaluation(genome.id, genome.name, 0, 1.42, Metrics(sharpe=1.1, trades=24))
    return genome, ev, journal


def test_llm_offspring_are_parsed_validated_and_reparented():
    claude = FakeClaude(_payload(3))
    breeder = LLMBreeder(claude, verbose=False)
    result = breeder.breed([_elite()], 3, generation=4, window="2015-2022")
    assert len(result.genomes) == 3
    assert all(is_valid(g) for g in result.genomes)
    assert all(g.generation == 4 and g.origin == "llm" for g in result.genomes)
    assert result.analysis.startswith("The winners")
    assert breeder.lessons == ["oversold entries need a trend filter"]
    assert result.cost_usd == 0.01


def test_ids_are_assigned_locally_not_taken_from_the_model():
    payload = _payload(1)
    payload["genomes"][0]["id"] = "../../etc/passwd"
    claude = FakeClaude(payload)
    genome = LLMBreeder(claude, verbose=False).breed([_elite()], 1, generation=1).genomes[0]
    assert genome.id != "../../etc/passwd"
    assert len(genome.id) == 12


def test_invalid_offspring_are_dropped_not_fatal():
    claude = FakeClaude(_payload(2, include_broken=True))
    result = LLMBreeder(claude, verbose=False).breed([_elite()], 4, generation=1)
    assert len(result.genomes) == 2
    assert result.rejected == 2


def test_api_failure_degrades_to_an_error_not_an_exception():
    claude = FakeClaude(None, fail=True)
    result = LLMBreeder(claude, verbose=False).breed([_elite()], 3, generation=1)
    assert result.genomes == [] and "rate limited" in result.error


def test_hybrid_backfills_with_mutation_when_claude_underdelivers():
    claude = FakeClaude(_payload(2))
    breeder = HybridBreeder(claude, rng=random.Random(0), llm_share=0.5, verbose=False)
    parent = _elite()[0]
    result = breeder.breed([_elite()], 10, generation=2, parent_pool=[parent])
    assert len(result.genomes) == 10
    assert {g.origin for g in result.genomes} >= {"llm"}
    assert all(is_valid(g) for g in result.genomes)


def test_hybrid_runs_without_claude_at_all():
    breeder = HybridBreeder(None, rng=random.Random(1), verbose=False)
    parent = _elite()[0]
    result = breeder.breed([_elite()], 8, generation=1, parent_pool=[parent])
    assert len(result.genomes) == 8 and all(is_valid(g) for g in result.genomes)


def test_llm_every_skips_generations():
    claude = FakeClaude(_payload(4))
    breeder = HybridBreeder(claude, rng=random.Random(2), llm_share=1.0, llm_every=3,
                            verbose=False)
    breeder.breed([_elite()], 4, generation=1, parent_pool=[_elite()[0]])
    assert claude.prompts == []
    breeder.breed([_elite()], 4, generation=3, parent_pool=[_elite()[0]])
    assert len(claude.prompts) == 1


def test_the_briefing_contains_the_evidence_claude_needs():
    genome, ev, journal = _elite()
    prompt = build_breeding_prompt(generation=7, elites=[(genome, ev, journal)],
                                   evals=[ev], history=[{"generation": 6, "best_score": 1.0,
                                                         "mean_score": 0.1, "best_name": "x"}],
                                   n_offspring=12, lessons=["trend filter helps"],
                                   window="train 2015..2022")
    assert "Dip Buyer" in prompt
    assert "rsi14 < 32" in prompt              # the rules themselves
    assert "SPY 2020-03-20" in prompt          # an actual trade
    assert "generation history" in prompt
    assert "trend filter helps" in prompt
    assert "12 new genomes" in prompt
    assert "cross_above(a, b)" in prompt       # the language reference


def test_schema_is_strict_enough_to_parse_blindly():
    schema = GENOME_JSON_SCHEMA
    assert schema["required"] == ["analysis", "lessons", "genomes"]
    item = schema["properties"]["genomes"]["items"]
    assert item["additionalProperties"] is False
    assert "entry_rules" in item["required"] and "risk" in item["required"]


def test_mutation_breeder_needs_no_parents():
    result = MutationBreeder(random.Random(3)).breed([], 5, generation=0)
    assert len(result.genomes) == 5 and all(is_valid(g) for g in result.genomes)
