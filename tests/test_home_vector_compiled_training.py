"""Compiled training, checkpoint portability and eager scientific evaluation."""
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.__main__ import main, train
from fly_connectome.home_vector.data import digest, verify_bundle
from fly_connectome.home_vector.model import HomeVectorNetwork
from test_home_vector_profile import prepared, arguments
from test_home_vector_gradient_assembly import four_population_circuit


def test_sequence_lengths_reuse_bounded_graphs_and_no_grad_stays_eager():
    torch._dynamo.reset()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    model = HomeVectorNetwork(four_population_circuit()).compile_recurrence(4, backend=backend)
    eager = HomeVectorNetwork(four_population_circuit())
    eager.load_state_dict(model.state_dict())
    for steps in (13, 18, 23, 28, 33, 38, 43, 48, 53, 58):
        model.zero_grad(set_to_none=True)
        x = torch.ones(2, steps, 2)
        pred, rates = model(x, .0025)
        expected_pred, expected_rates = eager(x, .0025)
        torch.testing.assert_close(pred, expected_pred)
        torch.testing.assert_close(rates, expected_rates)
        rates.square().mean().backward()
    # Initial zero state and differentiable continuation can specialize once
    # each; sequence lengths and short remainders should not create graphs.
    assert 1 <= len(graphs) <= 2
    count = len(graphs)
    with torch.no_grad():
        for steps in (11, 25, 47):
            for batch_size in (1, 3):
                x = torch.ones(batch_size, steps, 2)
                actual = model(x, .0025)
                expected = eager(x, .0025)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)
    assert len(graphs) == count


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compiled_train_validates_selects_and_reloads_eager_checkpoints(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch._dynamo.reset()
    source = prepared(tmp_path / 'source')
    hashes = {p.name: digest(p) for p in source.iterdir()}
    base = arguments(source, tmp_path / 'unused')
    base.device = device
    base.updates, base.validate_every = 2, 1
    base.max_seconds, base.benchmark = 600., False
    base.rewire_seed, base.no_recurrence = None, False
    for name, block in [('eager', 0), ('compiled', 4)]:
        args = copy.copy(base)
        args.output, args.compile_block = str(tmp_path / name), block
        train(args)
        manifest = verify_bundle(args.output)
        assert manifest['settings']['compile_block'] == block
        config = json.loads((tmp_path / name / 'training_config.json').read_text())
        assert config['compilation']['enabled'] == bool(block)
        assert config['compilation']['evaluation'] == 'eager'
        assert config['compilation']['block_steps'] == (block or None)
    assert hashes == {p.name: digest(p) for p in source.iterdir()}
    for checkpoint in ('last.pt', 'best.pt'):
        a, b = [torch.load(tmp_path / name / checkpoint, weights_only=False, map_location='cpu')
                for name in ('eager', 'compiled')]
        assert a['model'].keys() == b['model'].keys()
        assert a['update'] == b['update']
        for key in a['model']:
            torch.testing.assert_close(a['model'][key], b['model'][key], atol=2e-7, rtol=2e-5)
        # A normal eager model loads the selected compiled-training checkpoint.
        from test_home_vector_data import circuit_fixture
        eager = HomeVectorNetwork(circuit_fixture())
        eager.load_state_dict(b['model'], strict=True)
        assert eager._compiled_block is None
    for split in ('validation', 'test', 'challenge'):
        with np.load(tmp_path / 'eager' / f'{split}_predictions.npz') as a, np.load(tmp_path / 'compiled' / f'{split}_predictions.npz') as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_allclose(a[key], b[key], atol=2e-7, rtol=2e-5)
    a, b = [json.loads((tmp_path / name / 'metrics.json').read_text()) for name in ('eager', 'compiled')]
    assert a['selected_update'] == b['selected_update']
    assert a['numerical_validation']['passed'] == b['numerical_validation']['passed']
    for key in ('state_relative_rms_error', 'relative_rmse_change'):
        assert b['numerical_validation'][key] == pytest.approx(a['numerical_validation'][key], abs=1e-6)


@pytest.mark.parametrize('block', [-1, 1.5, True])
def test_invalid_compile_size_is_rejected_before_reading_source(block):
    from argparse import Namespace
    with pytest.raises(ValueError, match='compile block'):
        train(Namespace(compile_block=block))


def test_train_cli_exposes_compilation(monkeypatch, capsys):
    import fly_connectome.home_vector.__main__ as cli
    selected = []
    monkeypatch.setattr(cli, 'train', lambda args: selected.append(args) or dict(ok=True))
    main(['train', 'source', '--output', 'fresh', '--compile-block', '32', '--batch-size', '128'])
    assert selected[0].compile_block == 32
    assert selected[0].batch_size == 128
    assert json.loads(capsys.readouterr().out)['ok']
