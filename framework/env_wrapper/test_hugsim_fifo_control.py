import numpy as np

from framework.env_wrapper.hugsim_fifo_runner import execute_control_substeps


def test_fifo_control_uses_actual_simulator_time_step():
    class Env:
        dt = 0.25

        def step(self, action):
            assert action == {"acc": 0.5, "steer_rate": -0.1}
            return {}, 0.0, False, False, {"timestamp": self.dt}

    def controller(plan, info):
        assert plan.shape == (8, 2)
        assert info["dt"] == 0.25
        assert info["wheelbase"] == 2.7
        return 0.5, 0.1

    _, _, _, _, info = execute_control_substeps(
        env=Env(),
        plan=np.zeros((8, 2)),
        initial_info={"timestamp": 0.0, "dt": 0.5, "wheelbase": 2.7},
        traj2control=controller,
        substeps_per_rl_step=1,
    )
    assert info["hugsim_rl_step_delta_s"] == 0.25
