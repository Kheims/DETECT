"""Tests of the LLM context study scripts with the stub backend (no vLLM, no GPU).

Covers tool-call parsing (JSON and XML formats, malformed blocks, several
calls in one message), the agent loop (budget stop, answer stop, context
limit, malformed calls, final prompt fitting, tool output token counts), the
scoring path, the prompts of FACTS_ONLY, PAD and REPLAY, the C4 transcripts
REPLAY may use, the inputs of the prompt hashes, the context fitting that
never cuts the snippet, the static facts and type resolution of
scripts/llm/build_context_packages.py, the metrics, threshold tuning,
repository-clustered bootstrap and paired contrasts of
scripts/llm/eval_context_judge.py. On the real data (skipped when the context
packages are not built) it checks resume of every condition of
scripts/llm/llm_context_judge.py and an evaluation of the stub runs.

    .venv/bin/python -m unittest tests/test_context_judge.py
    .venv/bin/python -m pytest tests/test_context_judge.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.llm import build_context_packages as bcp  # noqa: E402
from scripts.llm import eval_context_judge as ecj  # noqa: E402
from scripts.llm import llm_context_judge as lcj  # noqa: E402
from scripts.llm.llm_judge import answer_token_ids, read_valid_rows, score_answer  # noqa: E402

CONFIG = PROJECT_ROOT / "config/experiments/context_study.yml"
PACKAGES = PROJECT_ROOT / "data/derived/context_packages/mlcq_context.jsonl"


class FakeTokenizer:
    """Word-level tokenizer with a minimal chat template (tools rendered in a
    <tools> block, tool messages in <tool_response> tags); decode(encode(x)) == x."""

    chat_template = "{{ messages }} {{ tools }}"
    eos_token_id = 0
    unk_token_id = None
    eos_token = "<|im_end|>"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<|im_end|>": 0}
        self.inv: dict[int, str] = {0: "<|im_end|>"}

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(p) for p in re.findall(r"<\|im_end\|>|\s|\w+|[^\w\s]", text)]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.inv[int(i)] for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.vocab.get(token)

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, **kwargs) -> str:
        out = ""
        if tools:
            out += "<tools>\n" + "\n".join(json.dumps(t) for t in tools) + "\n</tools>\n"
        for m in messages:
            if m["role"] == "tool":
                out += f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
            else:
                out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


def load_prompts() -> dict:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / "prompts/context_judge.yml").read_text())


BUDGETS = {"max_code_tokens": 500, "skeleton": 100, "facts": 100, "referenced_each": 50, "referenced_total": 100,
           "usages": 100, "replay_total": 300, "fit_scales": [1.0, 0.5, 0.25, 0.0]}


class ToolParsingTest(unittest.TestCase):
    def test_json_call(self):
        calls = lcj.parse_tool_calls('Let me check.\n<tool_call>\n{"name": "open_type", "arguments": {"name": "Order"}}'
                                     '\n</tool_call>')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "open_type")
        self.assertEqual(calls[0]["arguments"], {"name": "Order"})
        self.assertIsNone(calls[0]["error"])

    def test_xml_call(self):
        text = ("<tool_call>\n<function=read_method>\n<parameter=type_name>\nOrder\n</parameter>\n"
                "<parameter=method_name>\ngetTotal\n</parameter>\n</function>\n</tool_call>")
        calls = lcj.parse_tool_calls(text)
        self.assertEqual(calls[0]["name"], "read_method")
        self.assertEqual(calls[0]["arguments"], {"type_name": "Order", "method_name": "getTotal"})
        self.assertIsNone(calls[0]["error"])

    def test_arguments_as_json_string(self):
        calls = lcj.parse_tool_calls('<tool_call>{"name": "find_usages", "arguments": "{\\"name\\": \\"x\\"}"}'
                                     '</tool_call>')
        self.assertEqual(calls[0]["arguments"], {"name": "x"})

    def test_malformed_calls(self):
        bad_json = lcj.parse_tool_calls('<tool_call>{"name": "open_type", "arguments": {"name": }</tool_call>')
        self.assertIn("invalid JSON", bad_json[0]["error"])
        free_text = lcj.parse_tool_calls("<tool_call>open_type(Order)</tool_call>")
        self.assertIsNotNone(free_text[0]["error"])
        no_name = lcj.parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')
        self.assertIsNotNone(no_name[0]["error"])
        cut = lcj.parse_tool_calls("<tool_call>\n<function=open_type>\n<parameter=name>\nOrd")
        self.assertEqual(cut[0]["error"], "missing </function>")
        empty = lcj.parse_tool_calls("<tool_call>")
        self.assertIsNotNone(empty[0]["error"])

    def test_several_calls_in_one_message(self):
        text = ('<tool_call>{"name": "get_enclosing_class", "arguments": {}}</tool_call>\n'
                "<tool_call>\n<function=find_usages>\n<parameter=name>\nfoo\n</parameter>\n</function>\n</tool_call>\n"
                '<tool_call>{"name": "get_metrics"}</tool_call>')
        calls = lcj.parse_tool_calls(text)
        self.assertEqual([c["name"] for c in calls], ["get_enclosing_class", "find_usages", "get_metrics"])
        self.assertEqual(calls[1]["arguments"], {"name": "foo"})
        self.assertEqual(calls[2]["arguments"], {})
        self.assertTrue(all(c["error"] is None for c in calls))

    def test_no_call(self):
        self.assertEqual(lcj.parse_tool_calls("no"), [])


class FakeTools:
    """Tool executor that records calls."""

    def __init__(self, names):
        self.tool_names = names
        self.seen = []

    def call(self, name, arguments):
        self.seen.append((name, arguments))
        if name not in self.tool_names:
            return "unknown_tool", f"unknown {name}"
        return "ok", f"output of {name}"


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.prompts = load_prompts()
        self.tools_schema = self.prompts["agent"]["tools"]
        self.names = [t["function"]["name"] for t in self.tools_schema]
        self.yes = answer_token_ids(self.tok, ["yes", "Yes"])
        self.no = answer_token_ids(self.tok, ["no", "No"])
        self.builder = lcj.PromptBuilder(self.prompts, self.tok, dict(BUDGETS))
        self.acfg = {"max_tool_calls": 3, "max_turn_tokens": 64, "max_tool_output_tokens": 40,
                     "final_reserve_tokens": 16, "final_question_margin_tokens": 32, "message_overhead_tokens": 16}
        self.note = self.prompts["agent"]["messages"]["output_dropped"]

    def conversation(self, smell="is_feature_envy"):
        user = self.builder.user_message(self.prompts["agent"]["user_template"], smell, "void f() {}", False,
                                         max_tool_calls=self.acfg["max_tool_calls"])
        return lcj.Conversation(idx=1, smell=smell, level="method",
                                messages=[{"role": "system", "content": "S"}, {"role": "user", "content": user}],
                                tools=FakeTools(self.names), budget=self.acfg["max_tool_calls"], truncated=False,
                                code_tokens=3)

    def run_chunk(self, convs, policy, max_len=None, acfg=None):
        backend = lcj.StubBackend(self.tok, self.yes, self.no, policy)
        stats = lcj.run_agent_chunk(convs, backend, self.tok, self.tools_schema, {}, acfg or self.acfg,
                                    self.prompts["agent"], self.builder, max_len, [0], 20, self.yes, self.no)
        return backend, stats

    def test_budget_stop(self):
        conv = self.conversation()
        policy = lambda p: '<tool_call>{"name": "get_metrics", "arguments": {}}</tool_call>'  # noqa: E731
        backend, _ = self.run_chunk([conv], policy)
        self.assertEqual(conv.stop_reason, "budget")
        self.assertEqual(len(conv.calls), 3)
        self.assertEqual(conv.turns, 3)
        self.assertEqual(backend.calls, [1, 1, 1, 1])  # three turns, then the final question
        self.assertEqual(conv.messages[-1]["role"], "user")
        self.assertIn("You cannot call more tools", conv.messages[-1]["content"])
        self.assertFalse(math.isnan(conv.final_result["score"]))

    def test_calls_over_budget_in_one_message(self):
        conv = self.conversation()
        many = "".join('<tool_call>{"name": "get_metrics", "arguments": {}}</tool_call>' for _ in range(5))
        # the turn must be long enough to hold the five calls
        self.run_chunk([conv], lambda p: many, acfg={**self.acfg, "max_turn_tokens": 200})
        self.assertEqual(conv.turns, 1)
        self.assertEqual([c["status"] for c in conv.calls], ["ok"] * 3 + ["over_budget"] * 2)
        self.assertEqual(len(conv.tools.seen), 3)
        self.assertEqual(conv.stop_reason, "budget")
        tool_msgs = [m["content"] for m in conv.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 5)
        self.assertIn("budget of 3 tool calls", tool_msgs[-1])

    def test_call_cut_by_turn_length_is_malformed(self):
        conv = self.conversation()
        many = "".join('<tool_call>{"name": "get_metrics", "arguments": {}}</tool_call>' for _ in range(5))
        self.run_chunk([conv], lambda p: many if "<tool_response>" not in p else "no")
        self.assertEqual([c["status"] for c in conv.calls], ["ok", "ok", "malformed"])

    def test_answer_stops_and_lockstep(self):
        answering, calling = self.conversation(), self.conversation("is_long_method")

        def policy(prompt):
            if "long method" in prompt and prompt.count("<tool_response>") < 2:
                return '<tool_call>{"name": "get_enclosing_class", "arguments": {}}</tool_call>'
            return "no"

        backend, stats = self.run_chunk([answering, calling], policy)
        self.assertEqual(answering.stop_reason, "answered")
        self.assertEqual(answering.calls, [])
        self.assertEqual(calling.stop_reason, "answered")
        self.assertEqual(len(calling.calls), 2)
        # turn 1 holds both conversations, turns 2 and 3 only the one still calling, then one final call
        self.assertEqual(backend.calls, [2, 1, 1, 2])
        self.assertEqual(stats["generate_calls"], 4)
        for conv in (answering, calling):
            self.assertGreater(conv.prompt_tokens_total, conv.final_prompt_tokens)
            self.assertGreater(conv.seconds, -1e-9)

    def test_malformed_call_counts_against_budget(self):
        conv = self.conversation()
        self.run_chunk([conv], lambda p: "<tool_call>open_type(Order)</tool_call>" if "<tool_response>" not in p
                       else "yes")
        self.assertEqual(conv.calls[0]["status"], "malformed")
        self.assertEqual(conv.budget, 2)
        self.assertIn("could not be parsed", conv.messages[3]["content"])
        self.assertEqual(conv.tools.seen, [])

    def test_unknown_tool(self):
        conv = self.conversation()
        self.run_chunk([conv], lambda p: '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>'
                       if "<tool_response>" not in p else "no")
        self.assertEqual(conv.calls[0]["status"], "unknown_tool")

    def test_context_limit(self):
        conv = self.conversation()
        start = len(lcj.render_chat(self.tok, conv.messages, {}, self.tools_schema))
        final = self.prompts["agent"]["final_question"].format(question=self.prompts["smells"][conv.smell]["question"])
        reserve = max(self.acfg["final_reserve_tokens"],
                      self.builder.ntok(final) + self.acfg["final_question_margin_tokens"])
        max_len = start + self.acfg["max_turn_tokens"] + reserve + 5
        policy = lambda p: '<tool_call>{"name": "get_metrics", "arguments": {}}</tool_call>'  # noqa: E731
        self.run_chunk([conv], policy, max_len=max_len)
        self.assertEqual(conv.stop_reason, "context_limit")
        self.assertEqual(conv.turns, 1)
        self.assertLessEqual(conv.final_prompt_tokens + 1, max_len)
        self.assertIn("void f() {}", conv.messages[1]["content"])

    def test_fit_final_replaces_then_drops(self):
        conv = self.conversation()
        conv.messages += [{"role": "assistant", "content": "call " * 30},
                          {"role": "tool", "content": "long output " * 40},
                          {"role": "user", "content": "final question"}]
        render = lambda m: lcj.render_chat(self.tok, m, {}, None)  # noqa: E731
        base = len(render(conv.messages[:2] + conv.messages[-1:]))
        ids = lcj.fit_final(conv, render, base + 30, self.note)
        self.assertLessEqual(len(ids) + 1, base + 30)
        self.assertEqual(conv.final_fit["outputs_replaced"], 1)
        self.assertGreaterEqual(conv.final_fit["messages_dropped"], 1)
        self.assertIn("void f() {}", conv.messages[1]["content"])
        self.assertEqual(conv.messages[-1]["content"], "final question")

    def test_tool_outputs_for_replay(self):
        conv = self.conversation()
        policy = lambda p: '<tool_call>{"name": "get_metrics", "arguments": {}}</tool_call>'  # noqa: E731
        self.run_chunk([conv], policy)
        outs = lcj.tool_outputs(conv, self.note)
        self.assertEqual(len(outs), 3)
        self.assertEqual([o["text"] for o in outs], ["output of get_metrics"] * 3)
        rows, transcripts = lcj.agent_rows([conv], "h", self.note)
        self.assertEqual(transcripts[0]["tool_outputs"], outs)
        self.assertEqual(rows[0]["n_tool_ok"], 3)

    def test_output_tokens_after_cut(self):
        conv = self.conversation()

        class LongTools(FakeTools):
            def call(self, name, arguments):
                return "ok", "\n".join(f"line {k} of a long output" for k in range(60))

        conv.tools = LongTools(self.names)
        self.run_chunk([conv], lambda p: '<tool_call>{"name": "get_metrics", "arguments": {}}</tool_call>'
                       if "<tool_response>" not in p else "no")
        record = conv.calls[0]
        shown = self.builder.ntok(conv.tool_messages[0]["content"])
        self.assertTrue(record["output_truncated"])
        self.assertEqual(record["output_tokens"], shown)
        self.assertGreater(record["output_tokens_before_cut"], self.acfg["max_tool_output_tokens"])
        self.assertLess(record["output_tokens"], record["output_tokens_before_cut"])

    def test_scoring_path(self):
        conv = self.conversation()
        backend, _ = self.run_chunk([conv], lambda p: "yes")
        final_ids = lcj.render_chat(self.tok, conv.messages, {}, self.tools_schema)
        out = backend.generate([final_ids], 1, 20, None)[0]
        p_yes = math.exp(next(lp for tid, lp, _ in out["first_top"] if tid in self.yes))
        self.assertAlmostEqual(conv.final_result["score"], p_yes / (p_yes + (1 - p_yes)), places=9)
        self.assertEqual(conv.final_prompt_tokens, len(final_ids))
        direct = score_answer(out["first_top"], self.yes, self.no)
        self.assertAlmostEqual(direct["score"], conv.final_result["score"], places=12)


class SingleTurnPromptTest(unittest.TestCase):
    SNIPPET = "void f() {\n    order.getTotal();\n    order.getItems();\n}"

    def setUp(self):
        self.tok = FakeTokenizer()
        self.prompts = load_prompts()
        self.builder = lcj.PromptBuilder(self.prompts, self.tok, dict(BUDGETS))
        self.pkg = {"target": {"name": "f"}, "skeleton_kind": "enclosing_type",
                    "skeleton": "class Billing {\n" + "    int field;\n" * 40 + "}",
                    "facts_text": "Declared in Billing (40 fields, 1 methods); 0 parameters.\nAccesses to fields of its "
                                  "own class: 0",
                    "referenced_skeletons": [{"skeleton": "class Order {\n    double getTotal();\n}"}]}
        self.entries = {7: {"code_snippet": self.SNIPPET}}

    def build(self, condition, max_len=None, replays=None):
        return lcj.build_single_turn(condition, [(7, "is_feature_envy")], {7: "method"}, self.entries, {7: self.pkg},
                                     {7: None}, replays or {}, self.builder, self.tok, {}, max_len)[0]

    def test_snippet_identical_and_context_cut_to_fit(self):
        code_block = f"```java\n{self.SNIPPET}\n```"
        c0 = self.build("C0")
        c3 = self.build("C3")
        self.assertEqual(c3["context_scale"], 1.0)
        tight = self.build("C3", max_len=len(c3["ids"]) - 40)
        self.assertLess(tight["context_scale"], 1.0)
        self.assertLessEqual(len(tight["ids"]) + 1, len(c3["ids"]) - 40)
        for p in (c0, c3, tight):
            self.assertIn(code_block, self.tok.decode(p["ids"]))
        with self.assertRaises(SystemExit):
            self.build("C3", max_len=len(c0["ids"]) - 5)

    def test_facts_only_hides_code(self):
        p = self.build("FACTS_ONLY")
        text = self.tok.decode(p["ids"])
        self.assertNotIn("order.getTotal", text)
        self.assertIn("Declared in Billing", text)
        self.assertEqual(p["code_tokens"], 0)

    def test_pad_matches_c3_context_length(self):
        c3 = self.build("C3")
        pad = self.build("PAD")
        info = pad["context"]["pad"]
        self.assertEqual(info["target_tokens"], c3["context_tokens"])
        line_tokens = self.builder.ntok(self.prompts["pad"]["line"] + "\n")
        self.assertLessEqual(info["tokens"], info["target_tokens"])
        self.assertGreater(info["tokens"], info["target_tokens"] - line_tokens)
        text = self.tok.decode(pad["ids"])
        self.assertNotIn("class Billing", text)
        self.assertIn(self.prompts["pad"]["title"], text)
        self.assertIn(f"```java\n{self.SNIPPET}\n```", text)

    def test_replay_shows_outputs_in_order_without_agent_text(self):
        outputs = [{"name": "open_type", "arguments": {"name": "Order"}, "status": "ok", "text": "class Order {}",
                    "replaced": False},
                   {"name": "find_usages", "arguments": {"name": "zzz"}, "status": "empty", "text": "No line",
                    "replaced": False},
                   {"name": "get_metrics", "arguments": {}, "status": "over_budget", "text": "Not executed",
                    "replaced": False}]
        items = lcj.replay_items(outputs, ["ok", "empty"])
        self.assertEqual([o["name"] for o in items], ["open_type", "find_usages"])
        p = self.build("REPLAY", replays={(7, "is_feature_envy"): items})
        text = self.tok.decode(p["ids"])
        self.assertLess(text.index("[1] open_type(name=Order):"), text.index("[2] find_usages(name=zzz):"))
        self.assertNotIn("Not executed", text)
        self.assertEqual(p["context"]["replay"]["outputs"], 2)
        empty = self.build("REPLAY", replays={(7, "is_feature_envy"): []})
        self.assertIn("no tool output was retrieved", self.tok.decode(empty["ids"]))

    def test_load_replays_checks_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcripts.jsonl"
            path.write_text(json.dumps({"json_index": 3, "smell": "is_blob", "prompt_hash": "abc",
                                        "tool_outputs": [{"name": "get_metrics"}]}) + "\n")
            self.assertEqual(lcj.load_replays(Path(tmp), "abc"), {(3, "is_blob"): [{"name": "get_metrics"}]})
            with self.assertRaises(SystemExit):
                lcj.load_replays(Path(tmp), "other")

    def test_load_replays_keeps_scored_pairs_only(self):
        # (3, is_blob): a transcript left by a kill, then the one written with
        # its score; (4, is_blob): a transcript without a score
        rows = [{"json_index": 3, "smell": "is_blob", "prompt_hash": "abc", "tool_outputs": [{"name": "old"}]},
                {"json_index": 4, "smell": "is_blob", "prompt_hash": "abc", "tool_outputs": [{"name": "orphan"}]},
                {"json_index": 3, "smell": "is_blob", "prompt_hash": "abc", "tool_outputs": [{"name": "new"}]}]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "transcripts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            loaded = lcj.load_replays(Path(tmp), "abc", {(3, "is_blob")})
            self.assertEqual(loaded, {(3, "is_blob"): [{"name": "new"}]})


class FingerprintTest(unittest.TestCase):
    def setUp(self):
        import copy

        from mlcq_graphs.config import load_config

        self.copy = copy.deepcopy
        cfg = load_config(CONFIG, {})
        self.ccfg = cfg["context_study"]
        self.jcfg = self.ccfg["judge"]
        self.prompts = load_prompts()
        self.judge_cfg = self.jcfg["judges"]["qwen36_27b_fp8"]
        self.inputs = {"package_options": self.ccfg["packages"]["options"], "size_features_sha": "s1"}

    def fp(self, condition, jcfg=None, inputs=None):
        jcfg = jcfg or self.jcfg
        inputs = inputs or self.inputs
        c4 = lcj.fingerprint("C4", self.prompts, jcfg, self.judge_cfg, "p", "vllm", tool_inputs=inputs)
        if condition == "C4":
            return c4
        return lcj.fingerprint(condition, self.prompts, jcfg, self.judge_cfg, "p", "vllm",
                               c4 if condition == "REPLAY" else None, inputs)

    def test_replay_statuses_change_the_hash(self):
        other = self.copy(self.jcfg)
        other["replay"]["statuses"] = ["ok"]
        self.assertNotEqual(self.fp("REPLAY"), self.fp("REPLAY", jcfg=other))
        self.assertEqual(self.fp("C4"), self.fp("C4", jcfg=other))

    def test_tool_inputs_change_c4_and_replay(self):
        options = self.copy(self.inputs)
        options["package_options"]["header_chars"] += 1
        sizes = {**self.inputs, "size_features_sha": "s2"}
        for changed in (options, sizes):
            self.assertNotEqual(self.fp("C4"), self.fp("C4", inputs=changed))
            self.assertNotEqual(self.fp("REPLAY"), self.fp("REPLAY", inputs=changed))
            self.assertEqual(self.fp("C3"), self.fp("C3", inputs=changed))
        with self.assertRaises(ValueError):
            lcj.fingerprint("C4", self.prompts, self.jcfg, self.judge_cfg, "p", "vllm")

    def test_agent_margins_change_c4(self):
        other = self.copy(self.jcfg)
        other["agent"]["message_overhead_tokens"] += 1
        self.assertNotEqual(self.fp("C4"), self.fp("C4", jcfg=other))


class StaticAnalysisTest(unittest.TestCase):
    JAVA = b"""package com.shop;

import com.shop.model.Order;
import com.shop.model.Outer.Inner;
import java.util.List;
import com.shop.util.*;

public class Billing {
    private final Repo repo;
    private int count;
    public Billing(Repo repo) { this.repo = repo; }
    public double total(Order order, Customer c) {
        double sum = 0;
        for (Item it : order.getItems()) {
            sum += it.getPrice() * it.qty;
        }
        count++;
        this.repo.save(order);
        log(sum);
        org.apache.Util.check(order.id);
        Math.max(1, 2);
        ((Special) c).bonus();
        Runnable r = () -> helper();
        return sum + c.discount() + order.getTotal().value;
    }
    private void helper() {}
    public int getCount() { return count; }
    public void setCount(int count) { this.count = count; }
}
"""

    def test_method_facts(self):
        root = bcp.parse(self.JAVA).root_node
        target, iou = bcp.locate_target(root, "method", 12, 25)
        self.assertEqual(iou, 1.0)
        owners, anonymous = bcp.enclosing_types(target)
        facts = bcp.MethodFacts(target, self.JAVA, owners[0], owners[1:], anonymous)
        facts.visit(target.child_by_field_name("body"))
        d = facts.as_dict()
        self.assertEqual(d["own_field_accesses"], {"count": 1, "repo": 1})
        self.assertEqual(d["own_method_calls"], {"helper": 1})
        self.assertEqual(d["other_unqualified_calls"], {"log": 1})
        self.assertEqual(d["foreign"]["Order"]["members"], {"getItems()": 1, "id": 1, "getTotal()": 1})
        self.assertEqual(d["foreign"]["Order"]["via"], {"parameter order": 3})
        self.assertEqual(d["foreign"]["Item"]["members"], {"getPrice()": 1, "qty": 1})
        self.assertEqual(d["foreign"]["Repo"]["via"], {"field repo": 1})
        self.assertEqual(d["foreign"]["Special"]["via"], {"cast": 1})
        self.assertEqual(d["foreign"]["Util"]["members"], {"check()": 1})
        self.assertEqual(d["unresolved_receivers"], {"order.getTotal().value": 1})
        self.assertEqual(d["other_unqualified_names"], {})

    def test_class_facts_and_skeleton(self):
        root = bcp.parse(self.JAVA).root_node
        target, _ = bcp.locate_target(root, "class", 8, 30)
        facts = bcp.class_facts(target, self.JAVA, 8, 30)
        self.assertEqual((facts["fields"], facts["getters"], facts["setters"], facts["other_methods"],
                          facts["constructors"]), (2, 1, 1, 2, 1))
        opts = {"header_chars": 240, "signature_chars": 200, "modifiers_chars": 120, "field_type_chars": 80,
                "field_initializer_chars": 80, "max_enum_constants": 30}
        text, stats = bcp.type_skeleton(target, self.JAVA, "com.shop", opts, 200)
        self.assertIn("public double total(Order order, Customer c);", text)
        self.assertIn("private final Repo repo;", text)
        self.assertNotIn("sum +=", text)
        self.assertEqual(stats["methods"], 4)

    def test_type_resolution(self):
        root = bcp.parse(self.JAVA).root_node
        ctx = bcp.FileContext("src/com/shop/Billing.java", root, self.JAVA)
        types = {"Order": [["src/com/shop/model/Order.java", "com.shop.model"],
                           ["other/Order.java", "org.other"]],
                 "Outer": [["src/com/shop/model/Outer.java", "com.shop.model"]],
                 "Repo": [["src/com/shop/Repo.java", "com.shop"]],
                 "Helper": [["src/com/shop/util/Helper.java", "com.shop.util"]],
                 "Lonely": [["x/Lonely.java", "x"]],
                 "Twin": [["a/Twin.java", "a"], ["b/Twin.java", "b"]]}
        implicit = {"String"}
        r = lambda n: bcp.resolve_type(n, ctx, types, implicit)  # noqa: E731
        self.assertEqual((r("Order")["how"], r("Order")["path"]), ("import", "src/com/shop/model/Order.java"))
        self.assertEqual((r("Inner")["how"], r("Inner")["inner"]), ("import", "Outer.Inner"))
        self.assertEqual(r("Repo")["how"], "same_package")
        self.assertEqual(r("Helper")["how"], "wildcard_import")
        self.assertEqual(r("Lonely")["how"], "unique_name")
        self.assertEqual(r("Twin")["status"], "ambiguous")
        self.assertEqual(r("List")["status"], "external")
        self.assertEqual(r("String")["status"], "implicit")
        self.assertEqual(r("Nowhere")["status"], "unresolved")
        self.assertEqual(r("Billing")["how"], "same_file")
        self.assertEqual(r("org.other.Order")["path"], "other/Order.java")

    def test_proximity_order(self):
        files = ["b/x/Z.java", "a/y/B.java", "a/x/C.java", "a/x/A.java"]
        self.assertEqual(bcp.proximity_order(files, "a/x/A.java"),
                         ["a/x/A.java", "a/x/C.java", "a/y/B.java", "b/x/Z.java"])


class SelectionTest(unittest.TestCase):
    def test_agent_subset(self):
        rng = np.random.RandomState(1)
        indices = list(range(200))
        levels = {i: "method" if i < 120 else "class" for i in indices}
        labels = {i: [int(rng.rand() < 0.1), int(rng.rand() < 0.15), int(rng.rand() < 0.1), int(rng.rand() < 0.1)]
                  for i in indices}
        order = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
        smell_levels = {"is_feature_envy": "method", "is_long_method": "method", "is_blob": "class",
                        "is_data_class": "class"}
        sub = lcj.agent_subset(indices, levels, labels, order, smell_levels, 3, 0)
        self.assertEqual(sub, lcj.agent_subset(indices, levels, labels, order, smell_levels, 3, 0))
        for col, smell in enumerate(order):
            pos, neg = sub[smell]["positives"], sub[smell]["negatives"]
            self.assertEqual(len(neg), 3 * len(pos))
            self.assertTrue(all(labels[i][col] == 1 and levels[i] == smell_levels[smell] for i in pos))
            self.assertTrue(all(labels[i][col] == 0 and levels[i] == smell_levels[smell] for i in neg))
        capped = lcj.agent_subset(indices, levels, labels, order, smell_levels, 3, 0, positives_per_smell=2)
        self.assertTrue(all(len(v["positives"]) == 2 and len(v["negatives"]) == 6 for v in capped.values()))

    def test_shuffle_donors(self):
        indices = list(range(8))
        levels = {i: "method" if i < 6 else "class" for i in indices}
        packages = {i: {"skeleton": f"s{i}", "skeleton_kind": "enclosing_type" if i < 6 else "outer_type"}
                    for i in indices}
        packages[5]["skeleton"] = None
        repos = {0: "a", 1: "a", 2: "b", 3: "b", 4: "c", 5: "c", 6: "d", 7: "e"}
        donors = lcj.shuffle_donors(indices, levels, packages, repos, 0)
        self.assertEqual(donors, lcj.shuffle_donors(indices, levels, packages, repos, 0))
        self.assertIsNone(donors[5])
        for i, d in donors.items():
            if d is None:
                continue
            self.assertNotEqual(repos[i], repos[d])
            self.assertEqual(levels[i], levels[d])
            self.assertEqual(packages[i]["skeleton_kind"], packages[d]["skeleton_kind"])
            self.assertIsNotNone(packages[d]["skeleton"])


class MetricsTest(unittest.TestCase):
    def test_fast_metrics_match_sklearn(self):
        from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

        rng = np.random.default_rng(0)
        for _ in range(20):
            y = (rng.random(300) < 0.2).astype(int)
            s = np.round(rng.random(300), 2)  # ties
            self.assertAlmostEqual(ecj.fast_roc_auc(y, s), roc_auc_score(y, s), places=10)
            self.assertAlmostEqual(ecj.fast_average_precision(y, s), average_precision_score(y, s), places=10)
            pred = s >= 0.6
            self.assertAlmostEqual(ecj.mcc_binary(y, pred), matthews_corrcoef(y, pred), places=12)
        self.assertEqual(ecj.mcc_binary(np.array([0, 1]), np.array([False, False])), 0.0)

    def test_fast_tune_matches_cv(self):
        from mlcq_graphs.evaluation.cv import tune_thresholds

        rng = np.random.default_rng(1)
        for k in range(30):
            y = (rng.random(200) < 0.15).astype(np.int64)
            # scores on the grid, near it and far below it (very confident judges)
            s = np.where(rng.random(200) < 0.3, np.round(rng.random(200), 2), rng.random(200) ** (1 + k % 5))
            expected = tune_thresholds(y[:, None], s[:, None], 101)[0]
            self.assertEqual(ecj.fast_tune(y, s, 101), expected)
        self.assertEqual(ecj.fast_tune(np.zeros(5, dtype=np.int64), np.ones(5), 101), np.float32(0.5))

    @staticmethod
    def synthetic(n=300, seed=3):
        """Labels, repositories and scores: A label-informed, B = A, C = A ** 4
        (monotone), D noise."""
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.2).astype(np.int64)
        groups = np.array([f"r{k % 15}" for k in range(n)])
        a = 1.0 / (1.0 + np.exp(-(rng.normal(size=n) + 1.5 * y - 1.0)))
        return y, groups, {"A": a, "B": a.copy(), "C": a ** 4, "D": rng.random(n)}

    def test_contrasts_auc_are_paired(self):
        y, groups, scores = self.synthetic()
        draws, _ = ecj.cluster_draws(groups, 300, np.random.default_rng(0))
        contrasts = [("B", "A"), ("C", "A"), ("D", "A"), ("D", "C"), ("A", "missing")]
        out, paired = ecj.paired_bootstrap(y, scores, "A", draws, 0.95, contrasts)
        self.assertNotIn(("A", "missing", "roc_auc"), paired)
        for m in ecj.METRICS:
            self.assertEqual(paired[("B", "A", m)]["diff_ci"], [0.0, 0.0])
            # D - A on the same draws is the difference to the baseline A
            self.assertEqual(paired[("D", "A", m)]["diff_ci"], out["D"][f"{m}_diff_ci"])
            self.assertAlmostEqual(paired[("D", "C", m)]["diff"], out["D"][m] - out["C"][m], places=12)
        self.assertEqual(paired[("C", "A", "roc_auc")]["diff"], 0.0)
        self.assertEqual(paired[("C", "A", "roc_auc")]["diff_ci"], [0.0, 0.0])
        # recomputed draw by draw
        diffs = [ecj.fast_roc_auc(y[i], scores["D"][i]) - ecj.fast_roc_auc(y[i], scores["C"][i]) for i in draws
                 if y[i].min() != y[i].max()]
        self.assertEqual(paired[("D", "C", "roc_auc")]["diff_ci"], ecj.interval(np.array(diffs), 0.95))

    def test_contrasts_mcc_are_paired(self):
        y, groups, scores = self.synthetic()
        n = len(y)
        chunks = np.array_split(np.random.default_rng(5).permutation(n), 3)
        splits = []
        for seed in (42, 43):
            for k in range(3):
                test, val = chunks[k], chunks[(k + 1) % 3]
                train = np.setdiff1d(np.arange(n), np.r_[test, val])
                splits.append({"strategy": "kfold", "seed": seed, "fold": k, "train": train.tolist(),
                               "val": val.tolist(), "test": test.tolist()})
        smell = "is_feature_envy"
        ecfg = {"threshold_steps": 101, "mcc_threshold_modes": ["frozen", "retuned"],
                "bootstrap": {"draws": 200, "seed": 0, "level": 0.95}, "strategies": ["kfold"], "seeds": None,
                "folds_subset": None}
        rows, paired = ecj.mcc_scope("all", {smell: np.arange(n)}, {c: {smell: s} for c, s in scores.items()},
                                     {smell: y}, {smell: groups}, {"splits": splits}, ecfg, [smell], "A", 1.0, 0,
                                     [("B", "A"), ("D", "A"), ("D", "C")])
        by = {(r["contrast"], r["threshold"]): r for r in paired}
        row = {(r["condition"], r["threshold"]): r for r in rows}
        for mode in ("frozen", "retuned"):
            same = by[("B - A", mode)]
            self.assertEqual((same["diff"], same["pooled_diff"], same["pooled_diff_ci"]), (0.0, 0.0, [0.0, 0.0]))
            self.assertEqual(by[("D - A", mode)]["pooled_diff_ci"], row[("D", mode)]["mcc_pooled_diff_ci"])
            dc = by[("D - C", mode)]
            self.assertAlmostEqual(dc["diff"], row[("D", mode)]["mcc"] - row[("C", mode)]["mcc"], places=12)
            self.assertAlmostEqual(dc["pooled_diff"], row[("D", mode)]["mcc_pooled"] - row[("C", mode)]["mcc_pooled"],
                                   places=12)

    def test_cluster_draws(self):
        groups = np.array(["a", "a", "b", "c", "c", "c"])
        idx, weights = ecj.cluster_draws(groups, 50, np.random.default_rng(0))
        self.assertEqual(weights.shape, (50, 3))
        self.assertTrue(np.all(weights.sum(axis=1) == 3))
        for positions, w in zip(idx, weights):
            counts = {g: int(np.sum(groups[positions] == g)) for g in "abc"}
            self.assertEqual(counts, {"a": 2 * w[0], "b": w[1], "c": 3 * w[2]})


@unittest.skipUnless(PACKAGES.exists(), "context packages not built")
class EndToEndTest(unittest.TestCase):
    """Runs on the real data with the stub backend: every condition is scored
    on a smoke subset that includes the agent subset, a second run with the
    same command scores nothing, a run after a kill completes the missing
    pairs only; then the evaluation runs on the stub scores."""

    def overrides(self, tmp: str) -> list[str]:
        return [f"--context_study.judge.out_root={tmp}/scores", "--context_study.judge.smoke.n_per_level=40",
                "--context_study.judge.smoke.min_positives_per_smell=5",
                "--context_study.judge.smoke.include_agent_subset=true",
                "--context_study.judge.agent_subset.positives_per_smell=3",
                "--context_study.judge.chunk_size=16", "--context_study.judge.agent.chunk_size=5",
                f"--context_study.eval.out_dir={tmp}/cv", f"--context_study.eval.report_dir={tmp}/report",
                "--context_study.eval.missing=drop", "--context_study.eval.bootstrap.draws=200",
                "--context_study.eval.figure.enabled=false"]

    def run_condition(self, tmp: str, condition: str, tok: FakeTokenizer):
        argv = ["--config", str(CONFIG), "--condition", condition, "--backend", "stub"] + self.overrides(tmp)
        return lcj.main(argv, tokenizer=tok)

    def test_conditions_resume_and_eval(self):
        tok = FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            for condition in lcj.CONDITIONS:
                out = self.run_condition(tmp, condition, tok)
                scores = out / "scores.jsonl"
                rows = read_valid_rows(scores)
                self.assertGreater(len(rows), 0, condition)
                keys = [(r["json_index"], r["smell"]) for r in rows]
                self.assertEqual(len(keys), len(set(keys)))
                # same command again: nothing new
                self.run_condition(tmp, condition, tok)
                self.assertEqual(len(read_valid_rows(scores)), len(rows))
                # kill simulation: drop the last two rows and leave a partial line
                lines = scores.read_text().splitlines()
                scores.write_text("\n".join(lines[:-2]) + "\n" + lines[-2][: len(lines[-2]) // 2])
                self.run_condition(tmp, condition, tok)
                again = read_valid_rows(scores)
                self.assertEqual(sorted((r["json_index"], r["smell"]) for r in again), sorted(keys))
                applicable = [r for r in again if r["applicable"]]
                if condition == "C4":
                    self.assertTrue(all(r["n_tool_calls"] == 2 for r in applicable))
                    self.assertTrue(all(r["stop_reason"] == "answered" for r in applicable))
                    self.assertTrue(all(c["status"] in ("ok", "unavailable") for r in applicable
                                        for c in r["tool_calls"]))
                    self.assertTrue((out / "subset.json").exists())
                    meta = json.loads((out / "meta.json").read_text())
                    self.assertEqual(meta["totals"]["costs"]["queries"], len(applicable))
                if condition == "REPLAY":
                    c4 = {(r["json_index"], r["smell"]) for r in read_valid_rows(out.parent / "C4" / "scores.jsonl")}
                    self.assertEqual({(r["json_index"], r["smell"]) for r in again}, c4)
                    self.assertTrue(all(r["context"]["replay"]["outputs"] <= 2 for r in applicable))
            from mlcq_graphs.config import load_config, parse_cli_overrides

            cfg = load_config(CONFIG, parse_cli_overrides(self.overrides(tmp)))
            report = ecj.evaluate(cfg, "qwen36_27b_fp8")
            ranking = json.loads((report / "ranking.json").read_text())["rows"]
            self.assertEqual({r["condition"] for r in ranking if r["scope"] == "subset"}, set(lcj.CONDITIONS))
            coverage = json.loads((report / "coverage.json").read_text())["rows"]
            self.assertTrue(all(r["scored"] == r["requested"] for r in coverage if r["scope"] == "subset"))
            # the stub answers yes or no on the first token
            self.assertTrue(all(r["top_token_not_answer"] == 0 for r in coverage))
            # contrasts: C4 - REPLAY on the subset only, the others on both scopes
            contrasts = json.loads((report / "contrasts.json").read_text())["rows"]
            found = {(r["scope"], r["contrast"], r["metric"]) for r in contrasts}
            for scope, names in (("subset", ["C4 - REPLAY", "C3 - PAD", "C1 - SHUF", "C2 - FACTS_ONLY"]),
                                 ("all", ["C3 - PAD", "C1 - SHUF", "C2 - FACTS_ONLY"])):
                for name in names:
                    for m in ("roc_auc", "pr_auc", "mcc"):
                        self.assertIn((scope, name, m), found)
            self.assertNotIn(("all", "C4 - REPLAY", "roc_auc"), found)
            for r in contrasts:
                if r["metric"] != "mcc":
                    self.assertAlmostEqual(r["diff"], r["value_a"] - r["value_b"], places=12)
            # retuned MCC of C0 on the full scope equals the records of the shared protocol
            mcc = json.loads((report / "mcc.json").read_text())["rows"]
            records = json.loads((Path(tmp) / "cv" / "qwen36_27b_fp8_C0" / "records.json").read_text())
            for smell in ("is_feature_envy", "is_blob"):
                row = next(r for r in mcc if r["scope"] == "all" and r["smell"] == smell and r["condition"] == "C0"
                           and r["threshold"] == "retuned" and r["strategy"] == "kfold"
                           and r["score_temperature"] == 1.0)
                per_seed = {}
                for rec in records:
                    if rec["combo"]["split_strategy"] != "kfold":
                        continue
                    value = rec["per_label_tuned"][smell]["mcc"]
                    if value is not None:
                        per_seed.setdefault(rec["seed"], []).append(value)
                expected = float(np.mean([np.mean(v) for v in per_seed.values()]))
                self.assertAlmostEqual(row["mcc"], expected, places=12)


if __name__ == "__main__":
    unittest.main()
