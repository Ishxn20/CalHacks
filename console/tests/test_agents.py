import concurrent.futures as cf
import time

import pytest
from conftest import wait_for

pytest.importorskip("uagents")
import uagents.asgi  # noqa: E402

from scrub_arm.config import load_tray_config  # noqa: E402
from scrub_console.agents import AgentArm, AgentNetwork, Understand  # noqa: E402
from scrub_console.arm_client import ArmClient, ArmError  # noqa: E402
from scrub_console.bridge import Bridge  # noqa: E402
from scrub_console.intent import Intent, understand  # noqa: E402
from scrub_console.ledger import IN_FIELD, Ledger  # noqa: E402


class FakeArm:
    def __init__(self):
        self.calls = []

    def handoff(self, fields):
        self.calls.append(("handoff", fields))
        if fields.get("instrument") == "hammer":
            raise ArmError("no hammer", "I don't know the instrument 'hammer'", 400)
        return {"job": {"id": 1}, "instrument": "forceps_resano"}

    def command(self, name):
        self.calls.append((name, None))
        return {"status": name}


@pytest.fixture(scope="module")
def net():
    hops, arm = [], FakeArm()

    def fake_understand(text, context):
        if text == "slow":
            time.sleep(1.0)
        return Intent("handoff", f"On it: {text}", {"instrument": text, "ctx": context}, "claude")

    n = AgentNetwork(fake_understand, arm, look=lambda: ["forceps_resano", "scissors_potts"],
                     on_hop=lambda a, b, w: hops.append((a, b, w)), timeout_s=10)
    n.start()
    n.hops, n.fake_arm = hops, arm
    return n


def test_agents_are_real_uagents_with_stable_addresses(net):
    assert all(a.startswith("agent1") for a in net.addresses.values()) and len(set(net.addresses.values())) == 3
    assert uagents.asgi.HOST == "127.0.0.1"                 # never the 0.0.0.0 default


def test_an_utterance_travels_gateway_to_voice_agent_and_back(net):
    i = net.understand("resano please", "phase=idle")
    assert (i.kind, i.say, i.source) == ("handoff", "On it: resano please", "claude")
    assert i.fields == {"instrument": "resano please", "ctx": "phase=idle"}
    assert ("gateway", "voice", "Understand") in net.hops and ("voice", "gateway", "handoff") in net.hops


def test_arm_commands_go_through_the_arm_agent(net):
    assert net.arm_call("handoff", {"instrument": "resano"}) == {"job": {"id": 1}, "instrument": "forceps_resano"}
    assert net.arm_call("release") == {"status": "release"}
    assert net.fake_arm.calls[-2:] == [("handoff", {"instrument": "resano"}), ("release", None)]
    assert ("gateway", "arm", "handoff") in net.hops and ("arm", "gateway", "ok") in net.hops


def test_a_refusal_comes_back_as_the_same_speakable_error(net):
    with pytest.raises(ArmError) as e:
        net.arm_call("handoff", {"instrument": "hammer"})
    assert e.value.say == "I don't know the instrument 'hammer'" and e.value.status == 400
    assert ("arm", "gateway", "refused") in net.hops


def test_the_eyes_agent_reports_what_the_camera_sees(net):
    assert net.look() == ["forceps_resano", "scissors_potts"]


def test_a_slow_agent_times_out_instead_of_hanging(net):
    with pytest.raises(cf.TimeoutError):
        net._ask(Understand(id="", text="slow", context=""), 0.2)
    assert net.understand("fast", "").kind == "handoff"          # and the network is still healthy afterwards


def test_the_bridge_runs_a_real_handoff_through_the_agents(net, arm_server, tmp_path):
    """voice agent understands -> bridge -> arm agent commands the (simulated) arm -> ledger follows."""
    direct = ArmClient(arm_server.url, arm_server.token)
    tray = load_tray_config()
    # reuse the running agents, but give them the real arm and the real understand function
    net._arm = direct
    net._understand = lambda text, ctx: understand(text, direct.resolve, None, ctx, tray)
    bridge = Bridge(AgentArm(net, direct), Ledger.from_tray(tray), net.understand, sim=True)
    bridge.poll()
    r = bridge.handle_text("castroviejo, backhand")
    assert r["ok"] and r["intent"] == "handoff"
    wait_for(bridge, lambda: bridge.status.get("phase") == "presenting")
    assert bridge.handle_text("got it")["intent"] == "release"
    wait_for(bridge, lambda: any(i["status"] == IN_FIELD for i in bridge.ledger.snapshot()))
    assert ("gateway", "arm", "handoff") in net.hops and ("gateway", "arm", "release") in net.hops
