"""
L1++ — 詞典 VA 特徵 + Arousal 強度表面特徵 (E18)

在 lexicon.py 的 L1 十維特徵之外，加上兩組不需要任何標註的表面特徵：
1. Arousal intensity（15 維）：標點密度、程度副詞、身體反應、睡眠困擾、
   焦慮恐懼、壓力事件、低喚醒詞、字元重複、句長節奏、否定與轉折。
2. 新住民 domain cues（6 維）：語言障礙、證件身份、工作壓力、家庭分離、
   文化適應、經濟壓力。

設計原則：
- 全部特徵壓到 [0,1]，與 L1 一致，直接 concat 進回歸頭。
- 詞類特徵用「每 100 字密度」正規化（cap 後除 cap），避免 CVAT 長文天生比 CVAS 短句多命中。
- 不動 lexicon.py（凍結，實驗 4 復現用），本檔繼承其 LexiconFeaturizer。

可獨立執行驗證：
    python lexicon_intensity.py
"""
import re

from lexicon import LexiconFeaturizer, FEATURE_NAMES as L1_FEATURE_NAMES

# ---------------- 詞表（新住民反思 / 中文反思文本常見 cue） ----------------
DEGREE_ADVERBS = [
    "非常", "超級", "超", "太", "好", "很", "極度", "極了", "完全", "徹底",
    "十分", "相當", "特別", "格外", "簡直", "整個", "快要", "幾乎", "真的", "爆",
]
BODY_REACTION = [
    "發抖", "顫抖", "心跳", "心悸", "喘不過氣", "呼吸急促", "冒汗", "手抖",
    "頭痛", "頭暈", "胸悶", "胸口悶", "胸口痛", "想吐", "噁心", "發冷",
    "雞皮疙瘩", "手心冒汗", "全身僵硬", "腿軟", "血壓", "胃痛", "吃不下",
]
SLEEP_DISTURBANCE = [
    "睡不著", "失眠", "難以入睡", "半夜醒", "睡不好", "淺眠", "惡夢", "噩夢",
    "翻來覆去", "熬夜", "整夜", "睡眠",
]
ANXIETY_FEAR = [
    "焦慮", "緊張", "害怕", "恐懼", "恐慌", "不安", "擔心", "擔憂", "驚慌",
    "崩潰", "嚇", "慌張", "慌", "抓狂", "煩躁", "暴怒", "氣炸", "憤怒",
    "生氣", "恐怖", "壓抑", "絕望",
]
PRESSURE_EVENT = [
    "壓力", "逼", "催", "截止", "期限", "趕", "來不及", "緊急", "突然",
    "意外", "出事", "吵架", "衝突", "罵", "責備", "刁難", "羞辱", "歧視",
    "打發", "拒絕",
]
LOW_AROUSAL = [
    "平靜", "平淡", "安穩", "放鬆", "安心", "淡淡", "慢慢", "靜靜", "悠閒",
    "平和", "舒緩", "釋懷", "坦然", "安詳", "從容", "麻木", "疲憊", "疲倦",
    "無力", "提不起勁", "發呆", "空空的", "平常心", "習慣了",
]
NEGATION = ["不", "沒", "沒有", "無法", "不能", "別", "未", "難以"]
CONTRAST = ["但是", "可是", "卻", "然而", "不過", "偏偏", "明明", "沒想到", "反而", "居然", "竟然"]

# 新住民 domain-specific cues
CUE_LANGUAGE = [
    "聽不懂", "不會說", "說不出", "中文", "國語", "台語", "語言", "口音",
    "翻譯", "溝通", "表達", "詞不達意", "看不懂",
]
CUE_DOCUMENT = [
    "居留證", "身分證", "身份", "簽證", "護照", "證件", "文件", "辦理",
    "區公所", "移民署", "歸化", "國籍", "戶籍", "申請", "手續",
]
CUE_WORK = [
    "工作", "上班", "加班", "老闆", "同事", "面試", "找工作", "失業",
    "開除", "薪水", "薪資", "打工", "工廠", "職場", "值班",
]
CUE_FAMILY = [
    "家鄉", "娘家", "想家", "思念", "父母", "母親", "爸媽", "回不去",
    "離鄉", "越洋", "視訊", "家人", "故鄉", "回家過年", "隔著",
]
CUE_CULTURE = [
    "習俗", "文化", "習慣", "適應", "風俗", "節日", "拜拜", "飲食",
    "口味", "不習慣", "融入", "陌生", "外地", "外國人", "新住民",
]
CUE_FINANCIAL = [
    "錢", "經濟", "房租", "學費", "開銷", "生活費", "負擔", "貸款",
    "存款", "寄錢", "匯錢", "欠債", "繳", "付不出",
]

_SENT_SPLIT = re.compile(r"[。！？!?…;；\n]+")

# (名稱, 詞表, 每100字密度 cap)：cap 決定飽和點，全部特徵最後落在 [0,1]
_KEYWORD_FEATURES = [
    ("degree_adverb",   DEGREE_ADVERBS,    8.0),
    ("body_reaction",   BODY_REACTION,     4.0),
    ("sleep_disturb",   SLEEP_DISTURBANCE, 3.0),
    ("anxiety_fear",    ANXIETY_FEAR,      5.0),
    ("pressure_event",  PRESSURE_EVENT,    5.0),
    ("low_arousal",     LOW_AROUSAL,       4.0),
]
_KEYWORD_FEATURES_TAIL = [
    ("negation",        NEGATION,          10.0),
    ("contrast_marker", CONTRAST,          5.0),
]
_DOMAIN_FEATURES = [
    ("cue_language",  CUE_LANGUAGE,  4.0),
    ("cue_document",  CUE_DOCUMENT,  4.0),
    ("cue_work",      CUE_WORK,      4.0),
    ("cue_family",    CUE_FAMILY,    4.0),
    ("cue_culture",   CUE_CULTURE,   4.0),
    ("cue_financial", CUE_FINANCIAL, 4.0),
]

INTENSITY_NAMES = (
    ["exclam_density", "question_density", "ellipsis_density"]
    + [n for n, _, _ in _KEYWORD_FEATURES]
    + ["repeat_char_ratio", "sent_len_mean", "sent_len_std", "short_sent_ratio"]
    + [n for n, _, _ in _KEYWORD_FEATURES_TAIL]
    + [n for n, _, _ in _DOMAIN_FEATURES]
)
FEATURE_NAMES_INTENSITY = L1_FEATURE_NAMES + INTENSITY_NAMES
FEATURE_DIM_INTENSITY = len(FEATURE_NAMES_INTENSITY)  # 10 + 21 = 31


def _density(count, n_char, cap):
    """每 100 字密度，cap 後壓到 [0,1]。"""
    d = count * 100.0 / max(n_char, 1)
    return min(d, cap) / cap


def _kw_count(text, words):
    return sum(text.count(w) for w in words)


class L1IntensityFeaturizer(LexiconFeaturizer):
    """L1（10 維）+ arousal intensity / domain cues（21 維）= 31 維。"""

    def featurize(self, text):
        text = text or ""
        feats = super().featurize(text)  # L1 十維
        n = max(len(text), 1)

        # 標點密度
        feats.append(_density(text.count("！") + text.count("!"), n, 3.0))
        feats.append(_density(text.count("？") + text.count("?"), n, 3.0))
        feats.append(_density(text.count("…") + text.count("..."), n, 3.0))

        # 強度詞類
        for _, words, cap in _KEYWORD_FEATURES:
            feats.append(_density(_kw_count(text, words), n, cap))

        # 字元重複（啊啊啊 / ！！！之類；cap 0.1 避免一般疊詞「慢慢」就飽和）
        rep = sum(1 for a, b in zip(text, text[1:]) if a == b)
        feats.append(min(rep / max(n - 1, 1), 0.1) / 0.1)

        # 句長節奏
        sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        lens = [len(s) for s in sents] or [len(text)]
        mean_len = sum(lens) / len(lens)
        var = sum((x - mean_len) ** 2 for x in lens) / len(lens)
        feats.append(min(mean_len / 50.0, 1.0))
        feats.append(min(var ** 0.5 / 25.0, 1.0))
        feats.append(sum(1 for x in lens if x <= 10) / len(lens))

        # 否定 / 轉折
        for _, words, cap in _KEYWORD_FEATURES_TAIL:
            feats.append(_density(_kw_count(text, words), n, cap))

        # 新住民 domain cues
        for _, words, cap in _DOMAIN_FEATURES:
            feats.append(_density(_kw_count(text, words), n, cap))

        assert len(feats) == FEATURE_DIM_INTENSITY
        return feats


def _demo():
    fz = L1IntensityFeaturizer()
    print(f"詞典大小: {len(fz)}；特徵維度: {FEATURE_DIM_INTENSITY}")
    examples = [
        "我聽到那一刻簡直抓狂，胸口悶得喘不過氣！我躲在廁所裡發抖，氣到全身發冷。",
        "今天天氣很好，我慢慢走去市場，心情平靜，買了菜就回家了。",
        "去區公所辦居留證又被打發走，說文件不齊。排了三小時全白費，晚上翻來覆去睡不著。",
        "老闆突然說要加班趕出貨，我很緊張，怕來不及接小孩……薪水又還沒發，房租快付不出來了！",
    ]
    for t in examples:
        f = fz.featurize(t)
        print(f"\n「{t}」")
        pairs = list(zip(FEATURE_NAMES_INTENSITY, f))
        for i in range(0, len(pairs), 5):
            print("  " + "  ".join(f"{k}={v:.2f}" for k, v in pairs[i:i + 5]))


if __name__ == "__main__":
    _demo()
