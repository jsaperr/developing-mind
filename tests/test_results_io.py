"""Checks for src/brian2_stdp/results_io.py. Pure stdlib; no brian2."""
import numpy as np

from src.brian2_stdp.results_io import load_result, save_result


def test_roundtrip_plain_and_gzip_are_identical(tmp_path):
    obj = {'seed': 3, 'weight_trace': np.random.default_rng(0).random((5, 40)).round(4).tolist(),
           'label': 'x'}
    save_result(tmp_path / 'a.json', obj)
    save_result(tmp_path / 'a.json.gz', obj)
    assert load_result(tmp_path / 'a.json') == obj
    assert load_result(tmp_path / 'a.json.gz') == obj


def test_gzip_is_smaller_for_trace_like_data(tmp_path):
    obj = {'weight_trace': np.random.default_rng(1).random((60, 500)).round(4).tolist()}
    save_result(tmp_path / 'b.json', obj)
    save_result(tmp_path / 'b.json.gz', obj)
    assert (tmp_path / 'b.json.gz').stat().st_size < 0.6 * (tmp_path / 'b.json').stat().st_size


def test_load_falls_back_to_the_gz_sibling_when_only_it_exists(tmp_path):
    save_result(tmp_path / 'c.json.gz', {'k': 1})
    assert load_result(tmp_path / 'c.json') == {'k': 1}     # caller kept the old filename
