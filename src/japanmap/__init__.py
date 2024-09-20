import pickle
import typing
from itertools import chain, pairwise
from pathlib import Path

import numpy as np

NUM_PREF: typing.Final[int] = 47

pref_names = (
    "_ 北海道 青森県 岩手県 宮城県 秋田県 山形県 福島県 茨城県 栃木県 "
    "群馬県 埼玉県 千葉県 東京都 神奈川県 新潟県 富山県 石川県 福井県 山梨県 "
    "長野県 岐阜県 静岡県 愛知県 三重県 滋賀県 京都府 大阪府 兵庫県 奈良県 "
    "和歌山県 鳥取県 島根県 岡山県 広島県 山口県 徳島県 香川県 愛媛県 高知県 "
    "福岡県 佐賀県 長崎県 熊本県 大分県 宮崎県 鹿児島県 沖縄県"
).split()
pref_ = {s.rstrip("都道府県"): i for i, s in enumerate(pref_names)}
pref = {s: i for i, s in enumerate(pref_names)}
groups = {
    "北海道": [1],
    "東北": [2, 3, 4, 5, 6, 7],
    "関東": [8, 9, 10, 11, 12, 13, 14],
    "中部": [15, 16, 17, 18, 19, 20, 21, 22, 23],
    "近畿": [24, 25, 26, 27, 28, 29, 30],
    "中国": [31, 32, 33, 34, 35],
    "四国": [36, 37, 38, 39],
    "九州": [40, 41, 42, 43, 44, 45, 46, 47],
}


def picture(dic=None, rate=1):
    """ラスターデータ"""
    from cv2 import floodFill, imread  # noqa: PLC0415
    from PIL.ImageColor import getrgb  # noqa: PLC0415

    pos = [
        eval(s)
        for s in """\
        0 15,15 52,6 57,9 54,19 52,9 52,19 52,24
        52,34 49,31 47,31 47,34 52,36 47,36 47,37 47,24 37,31 34,32
        32,34 44,36 42,34 37,34 42,39 37,39 34,43 32,39 29,39 29,41
        27,39 31,44 29,44 19,38 12,42 22,39 17,41 11,44 22,46 22,44
        17,46 19,48 7,48 3,50 2,52 7,54 8,49 9,54 5,59 54,56""".split()
    ]
    p = imread(str(Path(__file__).parent / "japan.png"))
    if hasattr(dic, "items"):
        for k, v in dic.items():
            i = k if isinstance(k, int) else pref_code(k)
            if 1 <= i <= NUM_PREF:
                c = tuple(int(t * rate) for t in v) if isinstance(v, tuple) else getrgb(v)
                floodFill(p, None, (pos[i][0] * 10, pos[i][1] * 10), c)
    return p


def pref_code(s):
    """(頭に0がつかない)都道府県コード"""
    return pref_.get(s.rstrip("都道府県"), 0)


def get_data(move_hokkaido=False, move_okinawa=False, rough=False):  # noqa: FBT002
    """境界リストと県別の(隣接県,境界index)のリスト"""

    pkl_file = "japan0.16.pkl" if rough else "japan.pkl"
    with (Path(__file__).parent / pkl_file).open("rb") as fp:
        qp, qo = pickle.load(fp)
        qp = [list(p) for p in qp]
    if move_hokkaido:
        for i in qo[0][0][1][:-1]:
            qp[i][0] = [qp[i][0][0] - 10, qp[i][0][1] - 4.5]
    if move_okinawa:
        for i in qo[46][0][1][:-1]:
            qp[i][0] = [qp[i][0][0] + 4.5, qp[i][0][1] + 5]
    return qp, qo


def is_faced2sea(ip, qpqo=None):
    """県庁所在地を含むエリアが海に面するか"""
    assert 1 <= ip <= NUM_PREF, f"Must be 1 <= ip <= {NUM_PREF}"
    _, qo = qpqo or get_data()
    return any(i[0] == 0 for i in qo[ip - 1])


def is_sandwiched2sea(ip, qpqo=None):
    """県庁所在地を含むエリアが海に挟まれるか"""
    assert 1 <= ip <= NUM_PREF, f"Must be 1 <= ip <= {NUM_PREF}"
    _, qo = qpqo or get_data()
    return sum(i[0] == 0 for i in qo[ip - 1]) > 1


def adjacent(ip=None, qpqo=None):
    """県庁所在地を含むエリアが隣接する県コード"""
    if ip is None:
        return [(i, j) for i in range(1, 48) for j in adjacent(i, qpqo)]
    assert 1 <= ip <= NUM_PREF, f"Must be 1 <= ip <= {NUM_PREF}"
    _, qo = qpqo or get_data()
    return sorted([cd for cd, _ in qo[ip - 1] if cd])


def pref_points(qpqo=None, *, rough=False):
    """県の境界(index list)のリスト"""
    qp, qo = qpqo or get_data(move_hokkaido=True, move_okinawa=True, rough=rough)
    return [[qp[i][0] for _, ls in qo[k] for i in ls] for k in range(len(qo))]


def pref_map(ips, cols=None, width=1, qpqo=None, *, rough=False, tostr=False, ratio=(0.812, -1)):
    """ベクトルデータ(SVG)"""
    from IPython.display import SVG  # noqa: PLC0415

    assert all(1 <= ip <= NUM_PREF for ip in ips), f"Must be 1 <= ip <= {NUM_PREF}"
    if cols is None:
        cols = "red fuchsia purple navy blue teal aqua green lime olive yellow orange orangered maroon".split()
    elif isinstance(cols, str) and cols == "gray":
        cols = ["#%02x%02x%02x" % ((i * 18 + 32,) * 3) for i in [1, 8, 5, 10, 3, 0, 4, 7, 2, 9, 6]]
    pnts = pref_points(qpqo, rough=rough)
    pp = [[[i[0] * ratio[0], i[1] * ratio[1]] for i in pnts[ip - 1]] for ip in ips]
    ppp = np.array(list(chain(*pp)))
    mx, mn = np.nanmax(ppp, 0), np.nanmin(ppp, 0)
    mx = max(mx - mn)
    _cnv = lambda p: "M" + " ".join(["L%g,%g" % (x, y) for x, y in (p - mn) / mx])[1:] + " Z"  # noqa: E731, UP031
    s = "".join(f'<path fill="{cols[i % len(cols)]}" d="{_cnv(p)}"/>' for i, p in enumerate(pp))
    tpl = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 %d 1">%s</svg>'
    s = tpl % (width, s)
    return s if tostr else SVG(s)


def inflate(qp, qo, k, dif):
    for i, ls in qo[k]:
        if i == 0:
            continue
        for j in range(1, len(ls) - 1):
            dif[ls[j]] += ((qp[ls[j - 1]][0] + qp[ls[j + 1]][0]) / 2 - qp[ls[j]][0]) * 0.05


def trans_area(target, qpqo=None, niter=20, alpha=0.1):
    """
    Calculate positions which area close to a target.
    target: list of ratio
    move_hokkaido: move hokkaido
    move_okinawa: move okinawa
    niter: number of iteration
    alpha: ratio of change
    """
    qp, qo = qpqo or get_data(move_hokkaido=True, move_okinawa=True)
    qp = [[np.array(p[0]), p[1]] for p in qp]
    aa = [area(qp, qo, k) for k in range(len(qo))]
    assert len(aa) == len(target), "Must be same size."
    target = np.array(target)
    target = target / target.mean() * aa
    for _ in range(niter):
        pnts = pref_points((qp, qo))
        me = [np.mean(pp, 0) for pp in pnts]
        dif = np.zeros((len(qp), 2))
        for k, t in enumerate(target):
            inflate(qp, qo, k, dif)
            a = area(qp, qo, k)
            r = (t - a) / a * alpha
            for _, ls in qo[k]:
                zz = np.array([qp[j][0] for j in ls[:-1]]) - me[k]
                rd = np.sqrt((zz * zz).sum(1))
                if k == 0:
                    rd = 0.1 + 0.9 * rd / rd.max()
                for i, j in enumerate(ls[:-1]):
                    dif[j] += zz[i] * r / qp[j][1]
        for j in range(len(qp)):
            qp[j][0] += dif[j]
    return qp, qo


def area(qp, qo, k):
    """面積"""
    pp = pref_points((qp, qo))[k]
    pp.append(pp[0])
    return abs(sum((i[0] - j[0]) * (i[1] + j[1]) for i, j in pairwise(pp))) / 2


def distance(x1, y1, x2, y2):
    """緯度経度→距離(km)"""
    rp, rq = 6356.752, 6378.137  # 極半径、赤道半径(km)
    rx1, ry1, rx2, ry2 = np.radians([x1, y1, x2, y2])
    p1 = np.arctan(rp / rq * np.tan(ry1))
    p2 = np.arctan(rp / rq * np.tan(ry2))
    an = np.arccos(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(rx1 - rx2))
    ca = (np.sin(an) - an) * (np.sin(p1) + np.sin(p2)) ** 2 / np.cos(an / 2) ** 2
    cb = (np.sin(an) + an) * (np.sin(p1) - np.sin(p2)) ** 2 / np.sin(an / 2) ** 2
    return rq * (an + (rq - rp) / rq / 8 * (ca - cb))
