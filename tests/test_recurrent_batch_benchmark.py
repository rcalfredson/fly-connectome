import copy

import pytest

torch = pytest.importorskip("torch")
from examples.recurrent_batch_benchmark import SmallRecurrentModel, main, objective, statistics_for


def test_repeating_a_trajectory_preserves_mean_loss_and_parameter_gradients():
    torch.manual_seed(4)
    initial = SmallRecurrentModel(3, 4)
    x, y = torch.rand(1, 17, 3), torch.rand(1, 17, 2)
    results = []
    for batch_size in [1, 8]:
        model = copy.deepcopy(initial)
        loss = objective(model, x.repeat(batch_size, 1, 1), y.repeat(batch_size, 1, 1))
        loss.backward()
        results.append((loss.detach(), {n: p.grad for n, p in model.named_parameters()}))
    torch.testing.assert_close(results[0][0], results[1][0])
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], atol=1e-7, rtol=1e-5)


def test_statistics_distinguish_throughput_from_update_time():
    small = statistics_for(8, 101, [1., 2., 3.])
    big = statistics_for(32, 101, [3., 4., 5.])
    assert big["median_update_s"] > small["median_update_s"]
    assert big["trajectories_per_s"] > small["trajectories_per_s"]
    assert big["trajectory_transitions_per_s"] == 800
    assert big["seconds_per_1024_trajectory_passes"] == 128


def test_freestanding_cli_cpu_smoke_and_output_protection(tmp_path):
    output = tmp_path / "toy.json"
    args = ["--devices", "cpu", "--batches", "2", "4", "--steps", "9", "--inputs", "3",
            "--neurons", "4", "--warmup", "1", "--repeats", "2", "--output", str(output)]
    result = main(args)
    assert result["status"] == "complete"
    assert len(result["results"]) == 2
    assert all(len(r["update_seconds"]) == 2 for r in result["results"])
    with pytest.raises(FileExistsError):
        main(args)
