import json

from sim.reach_policy import ReachPolicy
from sim.train_reach import main


def test_pause_then_resume_restores_steps_and_replay_buffer(tmp_path):
    common = ["--her", "--eval-every", "300", "--eval-n", "2", "--learning-starts", "200", "--out-dir", str(tmp_path)]
    main(["--name", "t", "--steps", "600", "--stop-at", "300", *common])
    ck = tmp_path / "checkpoints" / "t"
    meta = json.loads((ck / "meta.json").read_text())
    assert meta["num_timesteps"] == 300 and meta["her"] and meta["eval_seed"] == 777 and meta["target_steps"] == 600
    assert (ck / "replay_buffer.pkl").exists() and (tmp_path / "reach_t.zip").exists()

    model = main(["--resume", str(ck), "--stop-at", "300", *common[1:]])   # loads, nothing left to do yet
    assert model.num_timesteps == 300 and model.replay_buffer.size() == 300

    model = main(["--resume", str(ck), *common[1:]])                        # continues to the 600 target
    assert model.num_timesteps == 600 and model.replay_buffer.size() == 600
    assert len((tmp_path / "reach_t_curve.csv").read_text().strip().splitlines()) == 1 + 2   # header + 300 + 600
    assert ReachPolicy(tmp_path / "reach_t.zip").model.num_timesteps == 600


def test_camera_guided_run_pauses_and_resumes(tmp_path):
    common = ["--eval-every", "300", "--eval-n", "2", "--learning-starts", "200", "--out-dir", str(tmp_path)]
    main(["--name", "v", "--env", "servo", "--steps", "600", "--stop-at", "300", *common])
    meta = json.loads((tmp_path / "checkpoints" / "v" / "meta.json").read_text())
    assert meta["env"] == "servo" and meta["num_timesteps"] == 300
    model = main(["--resume", str(tmp_path / "checkpoints" / "v"), *common])
    assert model.num_timesteps == 600 and model.replay_buffer.size() == 600
