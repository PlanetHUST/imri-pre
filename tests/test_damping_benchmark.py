"""End-to-end matching and checkpoint identity checks without detector workflow."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from physical_process.damping_benchmark import reference_source

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_damping_models', ROOT/'scripts/compare_damping_models.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_common_random_sources_and_global_rng_untouched():
    args=SimpleNamespace(input_json=None,seed=260529305,repeats=3,sources=5,lifetime_myr=.1)
    state=np.random.get_state()
    first=runner.build_cohort(args)
    second=runner.build_cohort(args)
    assert first==second
    assert np.array_equal(state[1],np.random.get_state()[1])
    for repeat in range(3):
        rows=[x for x in first if x.realization==repeat]
        assert sum(x.source_weight for x in rows)==pytest.approx(5.)
        assert len({x.seed for x in rows})==1
    assert len({x.seed for x in first})==3


def test_checkpoint_replay_and_input_changes(tmp_path):
    evaluator=runner.Evaluator(tmp_path,False,{'code':'test'},50000)
    source=reference_source()
    first,history=evaluator.pair(source,experiment='test',history=True)
    replay=runner.Evaluator(tmp_path,True,{'code':'test'},50000)
    second,history2=replay.pair(source,experiment='test',history=True)
    assert replay.reused==2 and replay.evaluated==0
    assert first==second and history==history2
    runner.csv_rows(tmp_path/'fresh.csv',first)
    runner.csv_rows(tmp_path/'resumed.csv',second)
    assert (tmp_path/'fresh.csv').read_bytes()==(tmp_path/'resumed.csv').read_bytes()
    assert first[0]['input_sha256']==first[1]['input_sha256']
    changed,_=replay.pair(replace(source,source_weight=2.),experiment='test',history=True)
    assert replay.evaluated==2
    assert changed[0]['checkpoint_key']!=first[0]['checkpoint_key']
    assert changed[0]['source_weight']==changed[1]['source_weight']==2.


def test_exact_input_loading_preserves_every_value(tmp_path):
    source=reference_source(seed=123,source_id='original-id',source_weight=17.)
    path=tmp_path/'inputs.json'
    path.write_text(json.dumps([asdict(source)]))
    assert runner.build_cohort(SimpleNamespace(input_json=path))==[source]
    path.write_text(json.dumps([asdict(source),asdict(source)]))
    with pytest.raises(ValueError,match='duplicate'):
        runner.build_cohort(SimpleNamespace(input_json=path))
