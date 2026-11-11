# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise source conversion, portable bundling and planning as one pipeline."""

import base64
import json

import pytest
from PIL import Image
from pydantic import TypeAdapter

from benchmarks.opencode_multimodal_benchmarks import agentx_prepare, agentx_source
from benchmarks.opencode_multimodal_benchmarks.agentx_media import read_image
from benchmarks.opencode_multimodal_benchmarks.bundle_agentx import bundle
from benchmarks.opencode_multimodal_benchmarks.canonical import fingerprint
from benchmarks.opencode_multimodal_benchmarks.data import sha256_file
from nemo_gym.data import TaskDataValidator
from nemo_gym.eval.eval_plan import PlanBenchmark, PlanGym, build_eval_plan
from resources_servers.opencode_multimodal_benchmarks.task_data import TaskData


@pytest.fixture
def source_release(tmp_path, monkeypatch):
    root = tmp_path / "release"
    (root / "files").mkdir(parents=True)
    records, references, tasks = {}, {}, {}
    for identifier, color in [("2", "red"), ("522", "blue")]:
        image = root / "files" / f"{identifier}.png"
        Image.new("RGB", (4, 4), color).save(image)
        record = {"dialogs": [{"role": "user", "content": "Is the color red or blue?"}], "files": [{"type": "image"}]}
        reference = {"query_type": "Objective", "final_answer": {"value": color, "justification": "visible color"}}
        records[identifier], references[identifier] = record, [reference]
        task = agentx_source.reference_metadata(identifier, record, reference)
        tasks[identifier] = {
            "reference_sha256": fingerprint(agentx_source.reference_identity(task)),
            "modality": "Image",
            "media": [[image.name, sha256_file(image)]],
        }
    for name, value in [("dataset.json", records), ("data.json", references)]:
        (root / name).write_text(json.dumps(value))
    manifest = {
        "samples": 2,
        "tasks": tasks,
        "source_repo": "synthetic",
        "source_revision": "fixture",
        "additional_generative_ids": [],
        "source_sha256": {name: sha256_file(root / name) for name in ("dataset.json", "data.json")},
    }
    monkeypatch.setattr(agentx_source, "manifest", lambda _: manifest)
    monkeypatch.setattr(agentx_prepare, "manifest", lambda _: manifest)
    monkeypatch.chdir(tmp_path)
    return root


def test_source_bundle_roundtrip_preserves_bytes_ids_and_strict_schema(source_release, tmp_path):
    output = agentx_prepare.prepare(source_root=str(source_release))
    original = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in original] == ["2", "522"]
    validator = TaskDataValidator("opencode_multimodal_benchmarks", TypeAdapter(TaskData), str(output))
    for index, row in enumerate(original):
        validator.validate_row(index, row)
        url = row["responses_create_params"]["input"][0]["content"][1]["image_url"]
        assert url.startswith("data:image/png;base64,")
        assert read_image(url)[0] == (source_release / "files" / f"{row['task_id']}.png").read_bytes()
        assert "visible color" not in json.dumps(row["responses_create_params"])
    assert validator.report.clean, validator.report.summary()
    portable = tmp_path / "portable"
    receipt = bundle(output, portable, sha256=sha256_file(output))
    prepared = agentx_prepare.prepare(
        dataset=str(portable / "agentx.jsonl"), sha256=receipt["dataset_sha256"], media_root=str(portable / "media")
    )
    assert [json.loads(line) for line in prepared.read_text().splitlines()] == original
    params = dict(
        benchmark=PlanBenchmark(name="agentx", config_paths=[]),
        taskset="visual_tasks",
        gym=PlanGym(version="test", commit=None),
    )
    assert build_eval_plan(original, **params) == build_eval_plan(original[::-1], **params)
    selected = build_eval_plan(original, selected_task_ids=["522"], **params)
    assert [task.task_id.task_id for task in selected.tasks] == ["522"]


def test_embedded_media_tampering_and_conflicting_task_id_are_rejected(source_release):
    rows = agentx_source.convert_source(source_release, source_release / "prepared")
    rows[0]["task_id"] = "wrong"
    with pytest.raises(ValueError, match="task_id"):
        agentx_prepare.validate_rows(rows)
    rows[0]["task_id"] = "2"
    first, second = [row["responses_create_params"]["input"][0]["content"][1] for row in rows]
    first["image_url"] = second["image_url"]
    with pytest.raises(ValueError, match="differs from public source"):
        agentx_prepare.validate_rows(rows)


@pytest.mark.parametrize(
    "value",
    [
        "data:image/png;base64,!bad!",
        "data:image/png,abc",
        "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
    ],
)
def test_embedded_media_must_be_valid_image_bytes(value):
    with pytest.raises((ValueError, OSError)):
        read_image(value)


@pytest.mark.parametrize(
    "mutation",
    ["append_answer", "extra_text", "extra_message", "instructions", "prompt", "previous_response_id", "conversation"],
)
def test_portable_bundle_rejects_modified_policy_prompt_even_with_matching_checksum(
    source_release, tmp_path, mutation
):
    output = agentx_prepare.prepare(source_root=str(source_release))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    params = rows[0]["responses_create_params"]
    leaked = "The reference answer is " + rows[0]["verifier_metadata"]["answer"]
    if mutation == "append_answer":
        params["input"][0]["content"][0]["text"] += "\n" + leaked
    elif mutation == "extra_text":
        params["input"][0]["content"].append({"type": "input_text", "text": leaked})
    elif mutation == "extra_message":
        params["input"].append({"role": "user", "content": [{"type": "input_text", "text": leaked}]})
    else:
        params[mutation] = leaked
    modified = tmp_path / "modified.jsonl"
    modified.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="canonical policy"):
        agentx_prepare.prepare(dataset=str(modified), sha256=sha256_file(modified))


def test_canonical_question_can_contain_reference_answer(source_release):
    rows = agentx_source.convert_source(source_release, source_release / "prepared")
    assert all(row["verifier_metadata"]["answer"] in row["verifier_metadata"]["question"] for row in rows)
    agentx_prepare.validate_rows(rows)


def test_portable_bundle_rejects_tool_definitions_with_reference_hints(source_release, tmp_path):
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    output = agentx_prepare.prepare(source_root=str(source_release))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    params = rows[0]["responses_create_params"]
    params["tools"] = [
        {
            "type": "function",
            "name": "helper",
            "description": "The reference answer is " + rows[0]["verifier_metadata"]["answer"],
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]
    # This is a valid model request: canonical preparation must reject it before inference.
    NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
    modified = tmp_path / "with-tools.jsonl"
    modified.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="cannot include task-supplied tools"):
        agentx_prepare.prepare(dataset=str(modified), sha256=sha256_file(modified))


@pytest.mark.parametrize("tools", [None, []])
def test_empty_tool_settings_do_not_change_canonical_bundle(source_release, tools):
    rows = agentx_source.convert_source(source_release, source_release / "prepared")
    rows[0]["responses_create_params"]["tools"] = tools
    agentx_prepare.validate_rows(rows)


def test_video_prompt_preserves_frame_timestamps_and_rejects_appended_answers(source_release):
    from benchmarks.opencode_multimodal_benchmarks.data import ANSWER_INSTRUCTION

    rows = agentx_source.convert_source(source_release, source_release / "prepared")
    task = rows[0]["verifier_metadata"]
    task.update(modality="Video", video_sampling={"fps": 25.0, "frame_count": 51, "indices": [25]})
    entry = agentx_prepare.manifest("AgentX")["tasks"][task["index"]]
    entry.update(modality="Video", reference_sha256=fingerprint(agentx_source.reference_identity(task)))
    text = rows[0]["responses_create_params"]["input"][0]["content"][0]
    text["text"] = (
        "This is a video:\nFrame 1 sampled at 1.00 seconds: <image>\n\n"
        "Question: Is the color red or blue?\n\n" + ANSWER_INSTRUCTION
    )
    agentx_prepare.validate_rows(rows)
    text["text"] += "\nThe reference answer is red"
    with pytest.raises(ValueError, match="canonical policy prompt mismatch"):
        agentx_prepare.validate_rows(rows)
