# flake8: noqa: S101
from japanmap import pref_map


def test_pref_map():
    actual = pref_map([11]).data
    expected = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 1 1"><path fill="red" d="M0.0157315,0.388464 L0.0224557,0.356562 L0,0.30912 L0,0.30912 L0.0372944,0.254112 L0.092209,0.252961 L0.116395,0.220441 L0.194794,0.194253 L0.210918,0.164611 L0.281489,0.157992 L0.299366,0.128351 L0.301469,0.0834569 L0.357435,0 L0.516337,0.0512253 L0.549169,0.0310805 L0.634812,0.0941048 L0.743356,0.0730967 L0.770229,0.0952559 L0.810188,0.0722334 L0.810188,0.0722334 L0.823526,0.0830927 L0.823526,0.0830927 L0.859728,0.19598 L0.895248,0.201735 L0.895248,0.201735 L0.998183,0.416421 L1,0.514685 L1,0.514685 L0.895481,0.48146 L0.881694,0.488942 L0.879942,0.515706 L0.831336,0.496137 L0.786119,0.50218 L0.766724,0.529519 L0.744641,0.523476 L0.710992,0.54765 L0.705617,0.529519 L0.684118,0.532973 L0.702053,0.520742 L0.69183,0.504554 L0.570463,0.53729 L0.552645,0.508511 L0.516388,0.501766 L0.493203,0.4577 L0.403382,0.456351 L0.296948,0.423526 L0.257515,0.395647 L0.201286,0.424875 L0.194226,0.443767 L0.194226,0.443767 L0.1511,0.459192 L0.120927,0.433251 L0.0830319,0.433426 L0.0567013,0.397493 L0.0157315,0.388464 Z"/></svg>'
    assert actual == expected
