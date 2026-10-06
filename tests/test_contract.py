import json

from monkeytrain import contract
from monkeytrain.ollama import modelfile
from monkeytrain.tools import TOOLS


def test_contract_is_deterministic_json():
    a = json.dumps(contract.build_contract(), ensure_ascii=False)
    b = json.dumps(contract.build_contract(), ensure_ascii=False)
    assert a == b
    assert json.loads(a)["version"] == contract.CONTRACT_VERSION


def test_contract_covers_tools_errors_and_parse_failures():
    c = contract.build_contract()
    assert c["tools"] == TOOLS
    assert {"read_file", "edit_file", "write_file", "glob", "grep"} <= {f["tool"] for f in c["fixtures"]["tools"]}
    assert sum(f["expected"].startswith("Error") for f in c["fixtures"]["tools"]) >= 6
    assert any(f["expected"]["errorCount"] for f in c["fixtures"]["parse"])
    assert c["fixtures"]["systemPrompt"]["expected"].startswith("You are Monkey")


def test_tool_fixtures_start_from_the_same_files():
    c = contract.build_contract()
    for f in c["fixtures"]["tools"]:
        assert f["files"] == contract.TOOL_FILES


def test_modelfile_uses_contract_sampling():
    mf = modelfile("./m.gguf")
    for key, value in contract.SAMPLING.items():
        assert f"PARAMETER {key} {value}" in mf
    for stop in contract.STOP:
        assert f"PARAMETER stop {stop}" in mf
