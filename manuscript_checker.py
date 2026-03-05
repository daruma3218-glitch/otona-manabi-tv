#!/usr/bin/env python3
from __future__ import annotations

"""
資料→原稿化ワークフローシステム - CLI版 (v2)

7ステップ:
  Step 1: 分野選定
  Step 2: 情報収集（自動選定3人 + 指標のプロ + 具体論のプロ）
  Step 3: 原稿化 (Opus 4.6) — 4章構成YouTube原稿
  Step 4: 批評フェーズ（批評家 + 評論家 + アドバイザー + コピーライター 並列）
  Step 5: 意見選定 (Opus 4.6) — コンセプトに沿う意見を取捨選択 + タイトル選定
  Step 6: リライト (Opus 4.6) — 選定意見反映
  Step 7: ファクトチェック (Sonnet + Web検索)

使い方:
    python manuscript_checker.py 資料.txt --title "仮タイトル" --purpose "趣旨"
    python manuscript_checker.py 資料.txt --title "仮タイトル" --purpose "趣旨" -o 結果.txt
    python manuscript_checker.py 資料.txt --title "仮タイトル" --purpose "趣旨" --no-search
    python manuscript_checker.py --resume abc12345
    python manuscript_checker.py --list-checkpoints
    python manuscript_checker.py --list-history
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import anthropic

# チェックポイント保存ディレクトリ
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# 履歴保存ディレクトリ
HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(exist_ok=True)
MAX_HISTORY = 50

# デフォルト設定
DEFAULT_MODEL = "claude-sonnet-4-6"
INTEGRATION_MODEL = "claude-opus-4-6"
EXPERT_MAX_TOKENS = 12000      # ディープリサーチ: 情報量拡大
REVIEWER_MAX_TOKENS = 6000     # 批評家も簡潔に
OPINION_MAX_TOKENS = 8000      # 意見選定
DRAFTING_MAX_TOKENS = 16000
FACT_CHECK_MAX_TOKENS = 16000
REWRITE_MAX_TOKENS = 16000
REFERENCE_LIST_MAX_TOKENS = 8000
MAX_CONTINUATIONS = 5
MAX_RETRIES = 12               # リトライ回数を十分に確保
RETRY_INITIAL_WAIT = 30
RETRY_MAX_WAIT = 180
STEP_COOLDOWN = 3              # ステップ間の最小クールダウン秒数
EXPERT_COOLDOWN = 2            # 専門家間のクールダウン秒数（短め）
NUM_EXPERTS = 3
PARALLEL_BATCH_SIZE = 5        # 全員同時並列実行

# Web検索ツール定義（ディープリサーチ用: 多段検索で深掘り）
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 10,
}

# ファクトチェック専用
FACT_CHECK_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 15,
}

# 参考文献リスト生成用
REFERENCE_LIST_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 10,
}

# ──────────────────────────────────────────────
# 固定専門家: 指標のプロ
# ──────────────────────────────────────────────
DATA_EXPERT = {
    "name": "指標のプロ",
    "role_description": "資料内の数値・データ・統計情報を検証し、原稿に差し込むべき具体的なデータや指標を提案する専門家。",
    "system_prompt": (
        "あなたは統計データ・指標分析の専門家です。"
        "資料に含まれる数値やデータを検証し、不足している重要な指標を特定して提案してください。"
        "公的統計、業界データ、国際比較データなど、原稿の説得力を高めるために差し込むべき"
        "具体的な数値・グラフ・指標を積極的に提案してください。"
        "データの出典は必ず明記し、最新の数値を使用してください。"
    ),
}

DATA_EXPERT_USER_PROMPT_WITH_SEARCH = """\
以下の資料を「指標・データの専門家」としてディープリサーチしてください。

━━━ リサーチ方法 ━━━
まず資料のテーマに関する基本的なデータを検索し、そこで見つかった情報をもとにさらに深掘り検索を繰り返してください。
1回の検索で終わらず、最低5回以上は検索を行い、多角的にデータを収集してください。
検索の切り口を変えて（日本語・異なるキーワード・関連分野・時系列変化など）幅広く探ってください。
━━━━━━━━━━━━━

以下の構成で回答してください:

1. **既存データの検証** (資料内の数値の正確性を検索で確認、誤りがあれば正しい数値と出典)
2. **最新の統計・数値データ** (上位8個、具体的な数値と出典URL付き。政府統計・業界レポート・学術論文など信頼性の高い情報源を優先)
3. **時系列の変化・トレンド** (過去3-5年の推移データ、変化率など)
4. **国際比較・他国データ** (日本と海外の比較データ、2-3カ国分)
5. **意外性のあるデータ** (視聴者が驚くような統計・数値、2-3個)
6. **データ活用の注意点** (2-3つ)

各データには必ず出典URLを付記してください。

【資料】
{material}
"""

DATA_EXPERT_USER_PROMPT_NO_SEARCH = """\
以下の資料を「指標・データの専門家」として簡潔に分析してください。

以下の構成で、要点のみ記載してください:

1. **既存データの検証** (正確性に問題があるもののみ指摘)
2. **差し込むべきデータ・指標** (上位5個、どのようなデータが必要か具体的に)
3. **データ活用の注意点** (2-3つ)

冗長な説明は不要、要点のみ。

【資料】
{material}
"""

# ──────────────────────────────────────────────
# 固定専門家: 具体論のプロ
# ──────────────────────────────────────────────
CASE_STUDY_EXPERT = {
    "name": "具体論のプロ",
    "role_description": "資料テーマに関連する実際の企業事例・成功/失敗事例・海外事例をリサーチし、原稿に差し込むべき具体的エピソードを提案する専門家。",
    "system_prompt": (
        "あなたは具体的事例・ケーススタディの専門家です。"
        "資料のテーマに関連する実際の事例（企業の取り組み、政策の成果、海外の類似事例など）を"
        "Web検索で調査し、原稿の説得力を高めるための具体的なエピソードを提案してください。"
        "賛成事例だけでなく、反論・失敗事例・対立する見解の根拠となる事例もバランスよく収集してください。"
        "各事例には出典を必ず明記してください。"
    ),
}

CASE_STUDY_USER_PROMPT_WITH_SEARCH = """\
以下の資料を「具体論の専門家」としてディープリサーチしてください。

━━━ リサーチ方法 ━━━
まず資料テーマの代表的な事例を検索し、そこから関連キーワードや企業名を拾ってさらに深掘り検索を繰り返してください。
1回の検索で終わらず、最低5回以上は検索を行い、多角的に事例を収集してください。
「成功事例→なぜ成功？」「失敗事例→何が原因？」「海外→日本との違いは？」と掘り下げてください。
━━━━━━━━━━━━━

以下の構成で回答してください:

1. **成功事例** (3-5件、企業名・具体的な取り組み・数値的成果を含む。「誰が・何を・どうした結果・どうなった」の形式)
2. **失敗・苦戦事例** (2-3件、なぜうまくいかなかったのか原因分析付き)
3. **海外の先進事例** (2-3件、日本との違い・学べるポイント)
4. **異業種・意外な分野の事例** (1-2件、視聴者に新しい視点を提供できるもの)
5. **対立する見解・批判的意見** (2-3件、根拠付き)
6. **原稿に差し込むべきベストエピソード** (上位5つ、インパクト順。出典URL付き)

各事例には必ず出典URLを付記してください。

【資料】
{material}
"""

CASE_STUDY_USER_PROMPT_NO_SEARCH = """\
以下の資料を「具体論の専門家」として分析してください。

以下の構成で、要点のみ記載してください:

1. **賛成・支持する事例** (資料内から2-3件、根拠を具体的に)
2. **反論・対立する視点** (2-3件、なぜ反対意見が存在しうるか)
3. **原稿に差し込むべきエピソード** (上位3つ)

各事例は「誰が・何を・どうした結果・どうなった」の形で簡潔に。

【資料】
{material}
"""

# ──────────────────────────────────────────────
# プロンプト定義
# ──────────────────────────────────────────────

# --- Step 1: 分野選定 ---
FIELD_SELECTION_SYSTEM = """\
あなたは資料分析の専門家です。
与えられた資料のテーマ・内容を分析し、視点チェックに最も適した3つの専門分野を選定してください。

以下のルールに従ってください:
1. 資料の主題に直接関連する分野を優先する
2. 3つの分野はそれぞれ異なる視点を提供できるものにする
3. 実務的な視点と学術的な視点のバランスを取る
4. 一般視聴者/読者への影響を評価できる分野を含める

回答はJSON形式で、以下の構造で返してください。JSONのみを返し、それ以外のテキストは含めないでください:
{
  "fields": [
    {
      "name": "分野名",
      "role_description": "この専門家の役割の説明（1-2文）",
      "system_prompt": "この専門家になりきるためのシステムプロンプト（200字程度）"
    }
  ]
}
"""

FIELD_SELECTION_USER = """\
以下の資料を分析し、視点チェックに最適な3つの専門分野を選定してください。

【資料】
{material}
"""

# --- Step 2: 専門家視点チェック ---
EXPERT_USER_PROMPT_WITH_SEARCH = """\
以下の資料を、あなたの専門分野の視点からディープリサーチしてください。

━━━ リサーチ方法 ━━━
あなたの専門分野ならではの切り口で、資料テーマを多角的に検索してください。
1回の検索で終わらず、最低5回以上は検索を行い、表面的な情報の裏にある構造的な問題や最新動向まで掘り下げてください。
「なぜそうなっているのか？」「今後どうなるのか？」「専門家はどう見ているのか？」を探ってください。
━━━━━━━━━━━━━

以下の構成で回答してください:

1. **資料の問題点・改善点** (重要なもの上位5つ、検索結果に基づく具体的な根拠付き)
2. **最新動向・状況変化** (資料作成時から変わった点、最新の研究・報告・ニュース)
3. **専門家・有識者の見解** (この分野の権威や機関がどう評価しているか、2-3件)
4. **欠落している重要な視点** (資料が見落としている論点、5つまで。なぜ重要かの理由付き)
5. **視聴者が知るべき背景知識** (原稿を深みのあるものにするための文脈情報、2-3個)
6. **原稿化の注意点** (2-3つ)

検索で見つけた情報には必ず出典URLを明記してください。

【資料】
{material}
"""

EXPERT_USER_PROMPT_NO_SEARCH = """\
以下の資料を、あなたの専門分野の視点から簡潔にチェックしてください。

以下の構成で、各項目2-3行で簡潔に回答してください:

1. **問題点・改善点** (重要なもの上位5つ、具体的な根拠付き)
2. **事実確認** (数値・データの正確性)
3. **欠落している視点** (3つまで)
4. **原稿化の注意点** (2-3つ)

冗長な説明は不要、要点のみ記載してください。

【資料】
{material}
"""

# --- Step 3: 原稿化 ---
DRAFTING_SYSTEM = """\
あなたは最高の映像監督 兼 ディレクターであり、プロのライターでもあります。
文章とリサーチ内容を整理して、YouTubeチャンネル「大人の学び直しTV」の読み上げ用の原稿を作成してください。
あくまで動画として最後までストーリーがわかりやすく繋がり、楽しく、正しく学べる内容を目指してください。

コンセプト:
視聴者にとって理解がしやすく、学びが深められる原稿がゴール。
偏りや潜在的なバイアスが排除された視点で、視聴者自身が答えを考えれる意見を出す。
"""

DRAFTING_USER_PROMPT = """\
以下の仮タイトル・趣旨・資料・専門家リサーチ結果を踏まえて、YouTube動画「大人の学び直しTV」の読み上げ用原稿を作成してください。

【仮タイトル】
{tentative_title}

【趣旨】
{purpose}

【元の資料】
{material}

{expert_reviews}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
以下のルールに必ず従ってください:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

まず、趣旨を踏まえて「この動画全体で何を伝えたいか」を1文で定義してから、
各章が何を主張し、なぜその順番で並んでいるのかという「主張軸」を先に設計してください。
主張軸の設計ルール:
  - 動画全体の主張（1文）を定める
  - 4章それぞれの役割・主張を1文ずつ定める（例: 第1章＝問題提起、第2章＝原因の深掘り、第3章＝解決策と事例、第4章＝未来の展望とまとめ）
  - 4章の流れが「起→承→転→結」のように論理的に繋がるようにする
  - 各章の主張は趣旨に紐づいており、趣旨から逸脱しないこと

主張軸を設計した上で、原稿の冒頭に以下の形式で明示してください:
━━━ 主張軸 ━━━
【動画全体の主張】（1文）
【第1章の主張】章タイトル: （1文）
【第2章の主張】章タイトル: （1文）
【第3章の主張】章タイトル: （1文）
【第4章の主張】章タイトル: （1文）
━━━━━━━━━━

この主張軸に忠実に、最後まで視聴したくなるような4章構成を作成してください。
・必ず4章に分けてください。5章以上にすることは禁止します。
・各章の内容は必ず上記の主張軸に沿って書いてください。主張軸から外れるエピソードや議論は入れないでください。

視聴者を引き付ける1文目から、この動画をみるとどういうベネフィットがあるのかという導入部分（300文字程度）、説得力と納得感のある本編の4章（8,000文字以上）、最後の章には読後感を高めるような示唆に富んだまとめを内包してください。

・本編の文字数は必ず日本語で数えて8,000文字以上にしてください。その際、AI（あなた）がLLMで参照するトークンの数ではなく、日本語の文字数でカウントするので注意してください。
全体が冗長になりすぎないようには意識してください。
あえて余計なことを伝えないというのも動画視聴者の理解を促進するためには大切なことです。

・導入部分には「そこで今回は、●●について解説します。最後までご覧いただければあなたの▲▲リテラシーまであがること間違いありません。 このチャンネルではそういった情報を配信しております。少しでも必要な情報だと感じたらどこのチャンネルかわからなくなる前に、ぜひチャンネル登録をよろしくお願いいたします。それでは早速やっていきましょう。」を必ずいれてください。この部分を変更することを禁止とします。

・章内の小見出しは無しです。大見出しが4つ、小見出しは0です。箇条書きは禁止とします。また演出などは不要です。

・各章のタイトルは12文字以内で簡潔に、続きを見たくなるようなものにしてください。各章のタイトルに 第1章 などの表記は必要ありません。

・ですます調でお願いします。

・読み上げ原稿なので出来る限り「平易な言葉遣い」で伝えることをしっかりと意識してください。言葉選びにまよったら常に平易で簡易な方を選択してください。ただ、幼稚になり過ぎないように気を付けてください。

・体裁として、概念や流れを先に説明して、そのあとに固有名詞や具体例を出して伝えることでわかりやすくなるよう意識してください。

・重要なキーワードを取りこぼしてはいけませんが、専門用語などを使ったあとには「●●とは〜」などとわかりやすく補足をしてください。

・適宜「実際に〜」「例えば〜」「●●によると〜」などと具体的な事例や数値を盛り込んで内容の信ぴょう性を高めてください。

・過去の出来事でも「でした。」「いました。」ばかりを多用せず、没入感と解説ストーリーをしっかりと意識して「です。」「ます。」を使用してください。

・話の「つながり」と「流れ」がわかりやすくなるようにかなり強く意識してください。そのために文頭には「そこで」「そのため」「こうした背景から」「だからこそ」「そんな中」「ここで注目したいのは」「さらに」「つまり」などの、話を前に進める接続詞を使用してください。
  ※「ただし」「とはいえ」「しかしながら」「もっとも」など話を後ろに戻す逆接表現は極力使わないでください。使う場合は1章につき最大1回までとし、直後に必ず主張を再強化してください。

・すべての文末で改行をし、行間を一行あけてください。

・一人称は「私」、二人称は「あなた」にしてください。

・くどくなりすぎない程度に、登場人物の思いや考えをセリフを言っているように「」で表現することでより理解しやすくなるように工夫してください。

・本編4章の合計が8,000文字以上になるよう、各章2,000文字以上を目安にしっかりとした分量で書いてください。各章では具体例、数値データ、エピソードを十分に盛り込み、説明を省略しないでください。短くまとめようとせず、視聴者が深く理解できる丁寧な解説を心がけてください。

・以下の言葉は使用禁止です。別の言葉に書き換えてください。「アキレス腱」「時限爆弾」「不確実性」「構造的」「ジレンマ」「明確に」「複合的な」「単なる」「ただし」「とはいえ」「しかしながら」「もっとも」

・大袈裟な比喩は使わないでください。

・各章は以下の流れで構成してください（話が行ったり来たりしないために重要です）:
  ① 主張を提示する（この章で何を伝えるか）
  ② 根拠・具体例・データで主張を裏付ける
  ③ 必要なら1回だけ別の視点に触れる（「一方で〜という見方もあります」程度）
  ④ 主張をさらに強化して章を締める
  ※この順序を守り、②と③を何度も行き来しないでください。話は常に前に進めてください。
"""

# --- Step 4: 批評フェーズ（4人のレビュアー） ---

REVIEW_CONCEPT = """\
コンセプト（最優先）:
視聴者にとって理解がしやすく、学びが深められる原稿がゴールです。
偏りや潜在的なバイアスが排除された視点で、視聴者自身が答えを考えれる意見を出せられるようにする。
すべての議論や批評を拾おうとするのではなく、あくまで意図やコンセプトが優先されます。

主張軸の遵守:
原稿冒頭に「主張軸」が定義されています。各章の改善提案は、この主張軸に沿ったものだけにしてください。
主張軸から逸脱するような新しい話題の追加や、章の役割を変えるような提案は避けてください。
改善提案は「各章の主張をより効果的に伝えるための改善」に限定してください。
"""

CRITIC_AGENT = {
    "name": "批評家",
    "system_prompt": (
        "あなたはYouTube教育コンテンツの批評家です。\n"
        "原稿の各章について、論理の飛躍、説得力不足の箇所、視聴者が離脱しそうなポイントを鋭く指摘してください。\n"
        "ただし、すべてを批判するのではなく、動画の意図やコンセプトを理解した上で、\n"
        "本当に改善すべき重要な点のみを指摘してください。\n"
        "良い点は良いと認めた上で、改善すべき箇所を具体的に述べてください。"
    ),
}

CRITIC_USER_PROMPT = """\
以下のYouTube動画原稿を批評してください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

{review_concept}

各章ごとに以下の観点で批評してください:
1. 論理の一貫性（飛躍や矛盾がないか）
2. 説得力（根拠は十分か、視聴者が納得できるか）
3. 視聴者の離脱リスク（退屈・難解・冗長な箇所）
4. 改善すべき最重要ポイント（各章1-2点に絞る）

【原稿】
{draft}
"""

COMMENTATOR_AGENT = {
    "name": "評論家（補足説明）",
    "system_prompt": (
        "あなたはYouTube教育コンテンツの評論家であり、補足説明の専門家です。\n"
        "原稿の各章について、視聴者が理解しづらい箇所、説明が不足している概念、\n"
        "追加の背景情報があると理解が深まるポイントを指摘してください。\n"
        "視聴者目線で「ここがわかりにくい」「ここに補足があると親切」という点を具体的に提案してください。"
    ),
}

COMMENTATOR_USER_PROMPT = """\
以下のYouTube動画原稿について、補足説明が必要な箇所を指摘してください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

{review_concept}

各章ごとに以下の観点で評論してください:
1. 視聴者が理解しづらい箇所（専門用語、前提知識が必要な部分）
2. 説明が不足している概念（もう少し丁寧に説明すべき箇所）
3. 補足があると理解が深まるポイント（具体例、数値、背景情報の追加提案）
4. 各章1-2点の補足提案に絞ってください

【原稿】
{draft}
"""

ADVISOR_AGENT = {
    "name": "アドバイザー",
    "system_prompt": (
        "あなたはYouTube動画コンテンツのアドバイザーです。\n"
        "原稿の各章について、「こうしたらもっと良くなる」という改善提案を具体的に提示してください。\n"
        "視聴者のエンゲージメント、学びの深さ、動画としての完成度を高める観点からアドバイスしてください。\n"
        "全体のコンセプトや意図を尊重した上で、実用的な改善策を提案してください。"
    ),
}

ADVISOR_USER_PROMPT = """\
以下のYouTube動画原稿について、改善アドバイスを提供してください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

{review_concept}

各章ごとに以下の観点でアドバイスしてください:
1. エンゲージメント向上（視聴者を引き付ける工夫）
2. 学びの深化（より深い理解につながる改善）
3. ストーリーの流れ（章間のつながり、全体の構成）
4. 各章1-2点の具体的改善提案に絞ってください

【原稿】
{draft}
"""

COPYWRITER_AGENT = {
    "name": "コピーライター",
    "system_prompt": (
        "あなたはYouTubeのサムネイルタイトル専門のコピーライターです。\n"
        "大人の学び直しTVのすあし社長らしい、人生をより賢く豊かに生きるためのヒント・視点を感じられる、知的で教養あふれる、\n"
        "過度に煽りすぎていないようなYouTubeタイトルを作成してください。\n"
        "クリック率を高めつつも、視聴者の信頼を損なわない品位あるタイトルを心がけてください。"
    ),
}

COPYWRITER_USER_PROMPT = """\
以下のYouTube動画原稿に対して、最適なYouTubeタイトルを10案作成してください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

タイトル作成のガイドライン:
- 大人の学び直しTVのすあし社長らしい、人生をより賢く豊かに生きるためのヒント・視点を感じられる、知的で教養あふれるタイトル
- 過度に煽り過ぎていない（品位を保つ）
- YouTube動画としてクリック率を高められるもの
- 視聴者の知的好奇心を刺激するもの

10案を番号付きで提示してください。各タイトルに簡単な意図説明（1行）を添えてください。

【原稿】
{draft}
"""

# 全レビュアー定義
REVIEWER_AGENTS = [CRITIC_AGENT, COMMENTATOR_AGENT, ADVISOR_AGENT, COPYWRITER_AGENT]

# --- Step 5: 意見選定 ---
OPINION_SELECTION_SYSTEM = """\
あなたは優秀な編集長です。
複数の批評家・評論家・アドバイザーからの意見と、コピーライターからのタイトル案を精査し、
原稿のコンセプトに最も沿った意見だけを取捨選択してください。

コンセプト:
視聴者にとって理解がしやすく、学びが深められる原稿がゴール。
偏りや潜在的なバイアスが排除された視点で、視聴者自身が答えを考えれる意見を出す。

すべての批評や提案を取り入れるのではなく、コンセプトに合致し、
原稿を本当に良くする意見だけを選んでください。
"""

OPINION_SELECTION_USER_PROMPT = """\
以下の原稿に対する4人のレビュアーの意見を精査し、採用すべき意見を選定してください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

【原稿】
{draft}

━━━━━━ レビュアーの意見 ━━━━━━
{review_results}

━━━━━━━━━━━━━━━━━━━━━━━━

以下の形式で出力してください:

## 採用するタイトル
コピーライターの10案から最も適切な1つを選び、その理由を1行で述べてください。

## 採用する改善意見
コンセプトに沿った改善意見のみを選び、以下の形式で列挙してください:
- 【出典: 批評家/評論家/アドバイザー】改善内容の要約
- ...

各章ごとに、本当に必要な改善のみを厳選してください（各章0-2点）。
コンセプトから外れる意見、原稿の軸をブレさせる意見は採用しないでください。

※重要: 原稿冒頭に「主張軸」が定義されています。以下の意見は必ず不採用としてください:
- 各章の主張軸から逸脱する新しい話題の追加
- 章の役割や論点を変えてしまう提案
- 動画全体の主張と矛盾する内容
採用する改善意見は「各章の主張をより効果的に伝えるための改善」に限定してください。

## リライト時の注意事項
リライターへの申し送り事項を2-3点で簡潔に。

※以下は必ず含めてください（固定事項）:
- 原稿の文字数（8,000文字以上）を維持すること。改善意見の反映時に既存の内容を削除・圧縮しないこと。
- 改善は「既存の文章を削って置き換える」のではなく「具体例の追加や説明の充実による加筆」で行うこと。
"""

# --- Step 6: リライト ---
REWRITE_SYSTEM = """\
あなたはプロの原稿リライターです。
編集長が選定した改善意見を原稿に反映し、最終版を完成させてください。

リライトのルール:
1. 編集長が採用した改善意見のみを反映する（それ以外は変更しない）
2. 原稿の4章構成、文体、トーンは維持する
3. 読み上げ原稿としての自然さを保つ
4. 改善を加えても全体のストーリーの流れを壊さない
5. 文字数は8,000文字以上を必ず維持する。リライト前の原稿より文字数が減ることは絶対に避ける。文字数が不足する場合は、具体例の追加、数値データの補足、背景説明の充実によって補うこと。
6. 改善意見の反映は「既存の文章を削って置き換える」のではなく、「既存の内容を維持したまま追加・修正する」形で行う。各章の内容は原則として維持し、削除は最小限にとどめる。
7. 原稿冒頭に定義された「主張軸」を遵守する。各章は主張軸で定義された役割・主張から逸脱してはならない。改善は主張軸の範囲内で行う。
8. 以下の言葉は使用禁止: 「アキレス腱」「時限爆弾」「不確実性」「構造的」「ジレンマ」「明確に」「複合的な」「単なる」「ただし」「とはいえ」「しかしながら」「もっとも」
9. 逆接表現（「一方で」等）は1章につき最大1回までとし、直後に必ず主張を再強化する。話を後ろに戻す表現を連続して使わないこと。各章の流れは「主張→根拠・事例→（1回だけ別視点）→主張の強化」の一方向で構成する。
"""

REWRITE_USER_PROMPT = """\
以下の原稿に対して、編集長が選定した改善意見を反映してリライトしてください。

【採用タイトル・改善意見】
{selected_opinions}

【リライト前の原稿】
{draft}

━━━ リライト時の最重要ルール ━━━
・リライト前の原稿と同等以上の文字数（8,000文字以上）を必ず維持してください。文字数が減ることは禁止です。
・改善意見を反映する際は、既存の内容を削るのではなく、具体例や数値の追加、説明の充実によって改善してください。
・各章の既存の内容は原則として維持し、追加・修正のみ行ってください。
・原稿冒頭の「主張軸」を必ず維持してください。各章の主張軸から逸脱する内容を追加しないでください。
  改善意見の反映後も、各章が主張軸で定義された役割・主張を忠実に果たしていることを確認してください。
━━━━━━━━━━━━━━━━━━

採用された改善意見を反映した最終版原稿を全文出力してください。省略や要約は禁止です。
原稿の冒頭に、採用されたタイトルを「■ タイトル: 〜」の形式で記載してください。
その後に原稿本文を出力してください。
"""

# --- Step 7: ファクトチェック ---
FACT_CHECK_SYSTEM = """\
あなたはファクトチェックの専門家です。
与えられた原稿に含まれる事実・数値・主張・統計データをWeb検索で1つずつ検証してください。

以下のルールに従ってください:
1. 原稿中の具体的な数値・統計・日付・人名・組織名・法律名などを重点的に検証する
2. 検証にはWeb検索を積極的に使用し、信頼性の高い情報源（政府機関、学術機関、公的統計など）を優先する
3. 誤りを発見した場合は正しい情報に修正する
4. 出典が不明確な主張には出典を追加する
5. 検証できなかった項目はその旨を明記する
6. 原稿の文体・構成・トーンは変更しない（事実の修正のみ行う）
7. 修正版原稿は元の原稿と同等の文字数を維持する。文章の簡潔化・圧縮・省略は行わない。誤りのある箇所のみ最小限に差し替え、それ以外の文章は一字一句変更しない。
"""

FACT_CHECK_USER_PROMPT = """\
以下の原稿に含まれる事実・数値・主張をWeb検索で検証し、必要に応じて修正してください。

以下の出力形式で回答してください:

## ファクトチェック結果
- 検証した項目と結果を箇条書き（各項目に✅正確 / ⚠️修正 / ❓未検証 を付記）
- 修正した場合は修正前→修正後を明記
- 出典URLを付記

## 修正版原稿
ファクトチェック結果を反映した修正版原稿を、元の原稿と同等の文字数で全文出力してください。
修正不要の場合も原稿をそのまま全文出力してください。

━━━ 重要 ━━━
・修正版原稿は元の原稿の文字数（8,000文字以上）を維持してください。文章の省略、要約、圧縮は禁止です。
・修正は事実の誤りの訂正のみ行い、文章の簡潔化や構成の変更は行わないでください。
・誤りのある箇所のみ最小限に修正し、それ以外の文章はそのまま維持してください。
━━━━━━━━━━

【検証対象の原稿】
{draft}
"""

# --- Step 7b: 参考文献・出典リスト生成 ---
REFERENCE_LIST_SYSTEM = """\
あなたは参考文献・出典リストの作成専門家です。
与えられた最終原稿・ファクトチェック結果・専門家リサーチ結果をもとに、
原稿中の主要な主張・事実・数値データについて、対応する出典・情報源を章ごとに整理した参考文献リストを作成してください。

以下のルールに従ってください:
1. 原稿の章構成（導入〜第4章）に合わせてグループ化する
2. 各エントリは「原稿中の主張/事実の要約」→「出典情報（URL含む）」→「検証ステータス」の形式にする
3. 専門家リサーチ結果やファクトチェック結果に含まれるURLを漏れなく拾い上げる
4. 同じ出典が複数の主張に使われている場合もすべて記載する（どの主張に対応するか明確にする）
5. 出典URLが見つからない項目はWeb検索で正確な出典を探し、見つからなければ「出典未特定」として記載する
6. 末尾に重複なしの全出典URL一覧を番号付きでまとめる
7. URLは必ずWeb検索で実在を確認してから記載する。推測や捏造は絶対に行わない
"""

REFERENCE_LIST_USER_PROMPT = """\
以下の最終原稿・ファクトチェック結果・専門家リサーチ結果をもとに、参考文献・出典リストを作成してください。

この形式に厳密に従ってください:

## 参考文献・出典リスト

### 導入・全体
- 【主張/事実】（原稿中の主張や数値の要約）
  出典: （出典名やURL）
  検証: ✅正確 / ⚠️修正済み / ❓未検証 / ➖出典未特定

### 第1章:（章タイトル）
- 【主張/事実】...
  出典: ...
  検証: ...

### 第2章:（章タイトル）
- ...

### 第3章:（章タイトル）
- ...

### 第4章:（章タイトル）
- ...

### 出典一覧（重複なし）
1. 出典名 URL
2. ...

━━━━━━━━━━━━━━━━━━━━━━━━

【最終原稿】
{final_draft}

【ファクトチェック結果】
{fact_report}

【専門家リサーチ結果（出典情報を含む）】
{expert_sources}
"""


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

def extract_text_from_response(response) -> str:
    texts = []
    for block in response.content:
        if hasattr(block, "text"):
            texts.append(block.text)
    return "".join(texts)


def get_retry_wait(retry_num: int, error=None) -> float:
    """指数バックオフでリトライ待ち時間を計算。"""
    if error and hasattr(error, "response") and error.response is not None:
        headers = getattr(error.response, "headers", {})
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after) + 5, RETRY_MAX_WAIT)
            except (ValueError, TypeError):
                pass
    return min(RETRY_INITIAL_WAIT * (retry_num + 1), RETRY_MAX_WAIT)


async def api_call_with_retry(client, label: str, **kwargs) -> object:
    """リトライ付きAPI呼び出し（CLI版）。"""
    last_error = None
    for retry in range(MAX_RETRIES):
        try:
            return await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=300,  # 5分タイムアウト
            )
        except asyncio.TimeoutError:
            last_error = Exception("API呼び出しが5分以上応答なし")
            wait_sec = min(RETRY_INITIAL_WAIT * (retry + 1), RETRY_MAX_WAIT)
            print(f"    ⏳ {label} API応答タイムアウト、{int(wait_sec)}秒後にリトライ... ({retry + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_sec)
            continue
        except anthropic.RateLimitError as e:
            last_error = e
            wait_sec = get_retry_wait(retry, e)
            print(f"    ⏳ {label} レート制限のため{int(wait_sec)}秒待機中... (リトライ {retry + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_sec)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            last_error = e
            wait_sec = min(RETRY_INITIAL_WAIT * (retry + 1), RETRY_MAX_WAIT)
            print(f"    ⏳ {label} APIエラー ({type(e).__name__})、{int(wait_sec)}秒後にリトライ... ({retry + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_sec)
        except anthropic.APIError as e:
            if hasattr(e, "status_code") and e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            last_error = e
            wait_sec = min(RETRY_INITIAL_WAIT * (retry + 1), RETRY_MAX_WAIT)
            print(f"    ⏳ {label} APIエラー、{int(wait_sec)}秒後にリトライ... ({retry + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_sec)
    raise Exception(f"{label} {MAX_RETRIES}回リトライしましたが失敗しました。最後のエラー: {last_error}")


# ──────────────────────────────────────────────
# チェックポイント管理
# ──────────────────────────────────────────────

def save_checkpoint(cp_id: str, data: dict) -> None:
    """チェックポイントをJSONファイルに保存"""
    cp_path = CHECKPOINT_DIR / f"{cp_id}.json"
    cp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(cp_id: str) -> dict | None:
    """チェックポイントを読み込む。存在しなければNone"""
    cp_path = CHECKPOINT_DIR / f"{cp_id}.json"
    if cp_path.exists():
        return json.loads(cp_path.read_text(encoding="utf-8"))
    return None


def list_checkpoints() -> list[dict]:
    """保存済みチェックポイント一覧を返す（新しい順）"""
    result = []
    for f in sorted(CHECKPOINT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "id": f.stem,
                "completed_step": data.get("completed_step", 0),
                "total_steps": 7,
                "material_preview": data.get("material", "")[:80],
                "title_preview": data.get("tentative_title", "")[:40],
                "purpose_preview": data.get("purpose", "")[:60],
                "timestamp": data.get("timestamp", ""),
                "error": data.get("error", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return result[:10]


# ──────────────────────────────────────────────
# 履歴管理
# ──────────────────────────────────────────────

def save_history(job_id: str, data: dict) -> None:
    """完了ジョブを履歴に保存。上限超過時は古いものを削除。"""
    data["job_id"] = job_id
    data["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    h_path = HISTORY_DIR / f"{job_id}.json"
    h_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(files) > MAX_HISTORY:
        files[0].unlink()
        files.pop(0)


def list_history_items() -> list[dict]:
    """履歴一覧を返す（新しい順）"""
    result = []
    for f in sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "job_id": data.get("job_id", f.stem),
                "completed_at": data.get("completed_at", ""),
                "title_preview": data.get("tentative_title", "")[:40],
                "purpose_preview": data.get("purpose", "")[:60],
                "material_preview": data.get("material", "")[:80],
                "total_elapsed": data.get("total_elapsed", 0),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return result[:MAX_HISTORY]


def print_header(text: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_progress(step: int, total: int, message: str) -> None:
    print(f"\n[{step}/{total}] {message}")


async def call_with_continuation(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    label: str = "",
    use_search: bool = False,
    search_tool: dict | None = None,
) -> str:
    full_text = ""
    current_messages = list(messages)
    tools = [search_tool or WEB_SEARCH_TOOL] if use_search else []

    for attempt in range(MAX_CONTINUATIONS + 1):
        kwargs = dict(
            model=model, max_tokens=max_tokens,
            system=system, messages=current_messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = await api_call_with_retry(client, label, **kwargs)

        chunk = extract_text_from_response(response)
        full_text += chunk

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "pause_turn" and attempt < MAX_CONTINUATIONS:
            print(f"    🔍 {label} 検索結果を処理中... ({attempt + 1}回目)")
            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": "続けてください。"},
            ]
            continue

        if response.stop_reason == "max_tokens" and attempt < MAX_CONTINUATIONS:
            print(f"    ⏳ {label} 出力が長いため続きを取得中... ({attempt + 1}回目)")
            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": "途切れてしまいました。直前の続きから、最後まで完成させてください。"},
            ]
        else:
            break

    return full_text


# ──────────────────────────────────────────────
# パイプライン関数
# ──────────────────────────────────────────────

async def select_expert_fields(client, material: str, model: str) -> list[dict]:
    print_progress(1, 7, "資料を分析し、最適な専門分野を選定中...")
    response = await client.messages.create(
        model=model, max_tokens=2048, system=FIELD_SELECTION_SYSTEM,
        messages=[{"role": "user", "content": FIELD_SELECTION_USER.format(material=material)}],
    )
    response_text = extract_text_from_response(response).strip()
    if "```" in response_text:
        lines = response_text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        response_text = "\n".join(json_lines)

    result = json.loads(response_text)
    fields = result["fields"][:NUM_EXPERTS]

    # 固定専門家「指標のプロ」「具体論のプロ」を追加
    fields.append(DATA_EXPERT)
    fields.append(CASE_STUDY_EXPERT)

    print("\n  選定された専門分野:")
    for i, field in enumerate(fields, 1):
        print(f"    {i}. {field['name']} - {field['role_description']}")
    return fields


async def run_expert_review(client, material: str, field: dict, index: int, model: str, use_search: bool = False) -> dict:
    field_name = field["name"]
    search_label = "🔍" if use_search else ""
    print(f"    ▶ 専門家{index + 1} ({field_name}) {search_label}... チェック中")
    start_time = time.time()

    # 固定専門家用の専用プロンプトを使い分け
    is_data_expert = (field["name"] == "指標のプロ")
    is_case_study_expert = (field["name"] == "具体論のプロ")
    if is_data_expert:
        if use_search:
            user_prompt = DATA_EXPERT_USER_PROMPT_WITH_SEARCH.format(material=material)
        else:
            user_prompt = DATA_EXPERT_USER_PROMPT_NO_SEARCH.format(material=material)
    elif is_case_study_expert:
        if use_search:
            user_prompt = CASE_STUDY_USER_PROMPT_WITH_SEARCH.format(material=material)
        else:
            user_prompt = CASE_STUDY_USER_PROMPT_NO_SEARCH.format(material=material)
    else:
        if use_search:
            user_prompt = EXPERT_USER_PROMPT_WITH_SEARCH.format(material=material)
        else:
            user_prompt = EXPERT_USER_PROMPT_NO_SEARCH.format(material=material)

    review_text = await call_with_continuation(
        client, model, field["system_prompt"],
        [{"role": "user", "content": user_prompt}],
        EXPERT_MAX_TOKENS, label=f"専門家{index + 1}({field_name})",
        use_search=use_search,
    )
    elapsed = time.time() - start_time
    print(f"    ✓ 専門家{index + 1} ({field_name}) ... 完了 ({elapsed:.1f}秒)")
    return {
        "index": index, "field_name": field_name,
        "role_description": field["role_description"],
        "review": review_text, "elapsed": elapsed,
    }


async def draft_manuscript(client, material: str, expert_results: list[dict], tentative_title: str, purpose: str) -> str:
    print_progress(3, 7, f"4章構成の原稿を作成中...（{INTEGRATION_MODEL}）")
    expert_reviews_text = ""
    for r in expert_results:
        expert_reviews_text += f"\n{'─' * 50}\n"
        expert_reviews_text += f"【専門家{r['index'] + 1}: {r['field_name']}の視点チェック】\n"
        expert_reviews_text += f"（役割: {r['role_description']}）\n\n"
        expert_reviews_text += r["review"] + "\n"

    return await call_with_continuation(
        client, INTEGRATION_MODEL, DRAFTING_SYSTEM,
        [{"role": "user", "content": DRAFTING_USER_PROMPT.format(
            material=material, expert_reviews=expert_reviews_text,
            tentative_title=tentative_title, purpose=purpose,
        )}],
        DRAFTING_MAX_TOKENS, label="原稿化エージェント(Opus)",
    )


async def run_single_reviewer(client, draft: str, agent: dict, index: int, tentative_title: str, purpose: str, model: str) -> dict:
    """1人のレビュアーによる批評"""
    agent_name = agent["name"]
    print(f"    ▶ レビュアー{index + 1} ({agent_name}) ... 批評中")
    start_time = time.time()

    # エージェントごとの専用プロンプトを選択
    prompt_map = {
        "批評家": CRITIC_USER_PROMPT,
        "評論家（補足説明）": COMMENTATOR_USER_PROMPT,
        "アドバイザー": ADVISOR_USER_PROMPT,
        "コピーライター": COPYWRITER_USER_PROMPT,
    }
    user_prompt_template = prompt_map.get(agent_name, CRITIC_USER_PROMPT)
    user_prompt = user_prompt_template.format(
        draft=draft,
        tentative_title=tentative_title,
        purpose=purpose,
        review_concept=REVIEW_CONCEPT,
    )

    review_text = await call_with_continuation(
        client, model, agent["system_prompt"],
        [{"role": "user", "content": user_prompt}],
        REVIEWER_MAX_TOKENS, label=f"レビュアー{index + 1}({agent_name})",
    )

    elapsed = time.time() - start_time
    print(f"    ✓ レビュアー{index + 1} ({agent_name}) ... 完了 ({elapsed:.1f}秒)")
    return {
        "index": index,
        "name": agent_name,
        "review": review_text,
        "elapsed": elapsed,
    }


async def run_review_phase(client, draft: str, tentative_title: str, purpose: str, model: str) -> list[dict]:
    """4人のレビュアーを並列実行"""
    print_progress(4, 7, "4人のレビュアーが原稿を批評中...")
    tasks = [
        run_single_reviewer(client, draft, agent, i, tentative_title, purpose, model)
        for i, agent in enumerate(REVIEWER_AGENTS)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    review_results = []
    for r in results:
        if isinstance(r, Exception):
            raise r
        review_results.append(r)
    return sorted(review_results, key=lambda r: r["index"])


async def select_opinions(client, draft: str, review_results: list[dict], tentative_title: str, purpose: str) -> str:
    """コンセプトに沿った意見を取捨選択 + タイトル選定"""
    print_progress(5, 7, f"意見を取捨選択中...（{INTEGRATION_MODEL}）")

    review_text = ""
    for r in review_results:
        review_text += f"\n{'━' * 50}\n"
        review_text += f"【{r['name']}の意見】\n\n"
        review_text += r["review"]
        review_text += "\n"

    return await call_with_continuation(
        client, INTEGRATION_MODEL, OPINION_SELECTION_SYSTEM,
        [{"role": "user", "content": OPINION_SELECTION_USER_PROMPT.format(
            draft=draft, review_results=review_text,
            tentative_title=tentative_title, purpose=purpose,
        )}],
        OPINION_MAX_TOKENS, label="意見選定(Opus)",
    )


async def rewrite_manuscript(client, draft: str, selected_opinions: str) -> str:
    """選定意見を反映してリライト"""
    print_progress(6, 7, f"リライト中...（{INTEGRATION_MODEL}）")

    return await call_with_continuation(
        client, INTEGRATION_MODEL, REWRITE_SYSTEM,
        [{"role": "user", "content": REWRITE_USER_PROMPT.format(
            draft=draft, selected_opinions=selected_opinions,
        )}],
        REWRITE_MAX_TOKENS, label="リライト(Opus)",
    )


async def fact_check_draft(client, draft: str) -> tuple[str, str]:
    """原稿をWeb検索でファクトチェックし、(修正版原稿, ファクトチェックレポート)を返す"""
    print_progress(7, 7, f"ファクトチェック中...（{DEFAULT_MODEL} + Web検索）")

    result = await call_with_continuation(
        client, DEFAULT_MODEL, FACT_CHECK_SYSTEM,
        [{"role": "user", "content": FACT_CHECK_USER_PROMPT.format(draft=draft)}],
        FACT_CHECK_MAX_TOKENS, label="ファクトチェック(Sonnet)",
        use_search=True, search_tool=FACT_CHECK_SEARCH_TOOL,
    )

    # ファクトチェック結果と修正版原稿を分離（複数パターンに対応）
    fact_report = ""
    corrected_draft = result
    split_markers = ["## 修正版原稿", "# 修正版原稿", "##修正版原稿", "#修正版原稿",
                     "## 修正済み原稿", "# 修正済み原稿", "## 修正後原稿", "## ファクトチェック修正版"]
    for marker in split_markers:
        if marker in result:
            parts = result.split(marker, 1)
            fact_report = parts[0].strip()
            corrected_draft = parts[1].strip()
            print(f"\n  ファクトチェック結果:\n{fact_report[:500]}{'...' if len(fact_report) > 500 else ''}")
            break

    return corrected_draft, fact_report


async def generate_reference_list(
    client, final_draft: str, fact_report: str,
    expert_results: list[dict],
) -> str:
    """最終原稿と情報源データから参考文献・出典リストを生成する"""
    print_progress(7, 7, "参考文献・出典リストを生成中...")

    # 専門家リサーチ結果からURL含む情報を組み立て
    expert_sources_text = ""
    for r in expert_results:
        expert_sources_text += f"\n--- {r.get('field_name', '')} ---\n"
        expert_sources_text += r.get("review", "")
        expert_sources_text += "\n"

    result = await call_with_continuation(
        client, DEFAULT_MODEL, REFERENCE_LIST_SYSTEM,
        [{"role": "user", "content": REFERENCE_LIST_USER_PROMPT.format(
            final_draft=final_draft,
            fact_report=fact_report if fact_report else "（ファクトチェック結果なし）",
            expert_sources=expert_sources_text,
        )}],
        REFERENCE_LIST_MAX_TOKENS, label="参考文献リスト生成",
        use_search=True, search_tool=REFERENCE_LIST_SEARCH_TOOL,
    )

    return result


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

async def main_async(
    material_path: str | None,
    tentative_title: str,
    purpose: str,
    output_path: str | None,
    model: str,
    use_search: bool = True,
    resume_id: str | None = None,
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("エラー: 環境変数 ANTHROPIC_API_KEY が設定されていません。")
        sys.exit(1)

    # チェックポイントから再開
    cp_id = resume_id or str(uuid.uuid4())[:8]
    cp_data = load_checkpoint(cp_id) if resume_id else None
    start_step = (cp_data.get("completed_step", 0) + 1) if cp_data else 1

    if cp_data:
        # チェックポイントからデータ復元
        material = cp_data["material"]
        tentative_title = cp_data.get("tentative_title", tentative_title)
        purpose = cp_data.get("purpose", purpose)
        model = cp_data.get("model", model)
        use_search = cp_data.get("use_search", use_search)
        fields = cp_data.get("fields")
        expert_results = cp_data.get("expert_results")
        draft = cp_data.get("draft")
        review_results = cp_data.get("review_results")
        selected_opinions = cp_data.get("selected_opinions")
        rewritten_draft = cp_data.get("rewritten_draft")
        print(f"\n  💾 チェックポイント {cp_id} からStep {start_step}より再開します")
    else:
        material_file = Path(material_path)
        if not material_file.exists():
            print(f"エラー: ファイルが見つかりません: {material_path}")
            sys.exit(1)
        material = material_file.read_text(encoding="utf-8")
        if not material.strip():
            print("エラー: 資料ファイルが空です。")
            sys.exit(1)
        fields = None
        expert_results = None
        draft = None
        review_results = None
        selected_opinions = None
        rewritten_draft = None

    print_header("資料→原稿化ワークフローシステム v2 (7ステップ)")
    if material_path and not cp_data:
        print(f"  資料ファイル: {material_path}")
    print(f"  仮タイトル: {tentative_title}")
    print(f"  趣旨: {purpose[:80]}{'...' if len(purpose) > 80 else ''}")
    print(f"  批評モデル: {model}")
    print(f"  原稿化/リライト: {INTEGRATION_MODEL}")
    print(f"  Web検索: {'有効' if use_search else '無効'}")
    print(f"  資料の長さ: {len(material):,}文字")
    if cp_data:
        print(f"  チェックポイント: {cp_id} (Step {start_step}から再開)")

    # チェックポイント共通データ
    def cp_common():
        return {
            "material": material, "tentative_title": tentative_title,
            "purpose": purpose,
            "model": model, "use_search": use_search,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    client = anthropic.AsyncAnthropic(api_key=api_key)
    total_start = time.time()

    try:
        # Step 1: 分野選定
        if start_step <= 1:
            fields = await select_expert_fields(client, material, model)
            save_checkpoint(cp_id, {
                "completed_step": 1, "fields": fields, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)
        else:
            print(f"\n  ✓ Step 1: 分野選定（復元済み）")

        # Step 2: 情報収集（並列実行）
        if start_step <= 2:
            num_total_experts = len(fields)
            search_label = "（Web検索あり）" if use_search else ""
            print_progress(2, 7, f"{num_total_experts}人の専門家が資料をリサーチ中...{search_label}")
            expert_results = []
            for batch_start in range(0, num_total_experts, PARALLEL_BATCH_SIZE):
                if batch_start > 0:
                    await asyncio.sleep(EXPERT_COOLDOWN)
                batch = fields[batch_start:batch_start + PARALLEL_BATCH_SIZE]
                batch_label = ", ".join(f["name"] for f in batch)
                print(f"    ── バッチ実行: {batch_label}")
                tasks = [
                    run_expert_review(client, material, field, batch_start + j, model, use_search=use_search)
                    for j, field in enumerate(batch)
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for br in batch_results:
                    if isinstance(br, Exception):
                        raise br
                    expert_results.append(br)
            expert_results = sorted(expert_results, key=lambda r: r["index"])
            save_checkpoint(cp_id, {
                "completed_step": 2, "fields": fields, "expert_results": expert_results,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)
        else:
            print(f"  ✓ Step 2: 情報収集（復元済み）")

        # Step 3: 原稿化
        if start_step <= 3:
            draft = await draft_manuscript(client, material, expert_results, tentative_title, purpose)
            save_checkpoint(cp_id, {
                "completed_step": 3, "fields": fields, "expert_results": expert_results,
                "draft": draft, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)
        else:
            print(f"  ✓ Step 3: 原稿化（復元済み）")

        # Step 4: 批評フェーズ
        if start_step <= 4:
            review_results = await run_review_phase(client, draft, tentative_title, purpose, model)
            save_checkpoint(cp_id, {
                "completed_step": 4, "fields": fields, "expert_results": expert_results,
                "draft": draft, "review_results": review_results,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)
        else:
            print(f"  ✓ Step 4: 批評フェーズ（復元済み）")

        # Step 5: 意見選定
        if start_step <= 5:
            selected_opinions = await select_opinions(client, draft, review_results, tentative_title, purpose)
            save_checkpoint(cp_id, {
                "completed_step": 5, "fields": fields, "expert_results": expert_results,
                "draft": draft, "review_results": review_results,
                "selected_opinions": selected_opinions,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)
        else:
            print(f"  ✓ Step 5: 意見選定（復元済み）")

        # Step 6: リライト
        if start_step <= 6:
            rewritten_draft = await rewrite_manuscript(client, draft, selected_opinions)
            save_checkpoint(cp_id, {
                "completed_step": 6, "fields": fields, "expert_results": expert_results,
                "draft": draft, "review_results": review_results,
                "selected_opinions": selected_opinions, "rewritten_draft": rewritten_draft,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)
        else:
            print(f"  ✓ Step 6: リライト（復元済み）")

        # Step 7: ファクトチェック
        if start_step <= 7:
            final, fact_report = await fact_check_draft(client, rewritten_draft)

        # Step 7b: 参考文献・出典リスト生成
        reference_list = ""
        try:
            reference_list = await generate_reference_list(
                client, final, fact_report, expert_results
            )
        except Exception as ref_err:
            reference_list = f"（参考文献リストの生成に失敗しました: {ref_err}）"
            print(f"\n  ⚠ 参考文献リスト生成エラー: {ref_err}（原稿自体は正常に完成しています）")

        total_elapsed = time.time() - total_start

        # 履歴に保存
        save_history(cp_id, {
            "material": material,
            "tentative_title": tentative_title,
            "purpose": purpose,
            "fields": fields,
            "expert_results": expert_results,
            "draft": draft,
            "review_results": review_results,
            "selected_opinions": selected_opinions,
            "rewritten_draft": rewritten_draft,
            "final": final,
            "fact_report": fact_report,
            "reference_list": reference_list,
            "total_elapsed": round(total_elapsed, 1),
            "model": model,
            "use_search": use_search,
        })

        # 完了 → チェックポイント削除
        cp_path = CHECKPOINT_DIR / f"{cp_id}.json"
        if cp_path.exists():
            cp_path.unlink()

        # 出力
        sep = "=" * 70
        output_parts = [sep, " 資料→原稿化ワークフロー結果 (v2 / 7ステップ)", sep]

        output_parts.append(f"\n■ 仮タイトル: {tentative_title}")
        output_parts.append(f"■ 趣旨: {purpose}")

        output_parts.append("\n■ 選定された専門分野:")
        for i, field in enumerate(fields, 1):
            output_parts.append(f"  {i}. {field['name']} - {field['role_description']}")

        for r in expert_results:
            output_parts.append(f"\n{sep}")
            output_parts.append(f" 専門家{r['index'] + 1}: {r['field_name']}の視点チェック")
            output_parts.append(sep)
            output_parts.append(r["review"])

        output_parts.append(f"\n{sep}")
        output_parts.append(" 中間原稿 (Step 3)")
        output_parts.append(sep)
        output_parts.append(draft)

        for rv in review_results:
            output_parts.append(f"\n{sep}")
            output_parts.append(f" {rv['name']}の批評 (Step 4)")
            output_parts.append(sep)
            output_parts.append(rv["review"])

        output_parts.append(f"\n{sep}")
        output_parts.append(" 意見選定結果 (Step 5)")
        output_parts.append(sep)
        output_parts.append(selected_opinions)

        output_parts.append(f"\n{sep}")
        output_parts.append(" リライト済み原稿 (Step 6)")
        output_parts.append(sep)
        output_parts.append(rewritten_draft)

        output_parts.append(f"\n{sep}")
        output_parts.append(" 最終原稿 - ファクトチェック済み (Step 7)")
        output_parts.append(sep)
        output_parts.append(final)

        if reference_list:
            output_parts.append(f"\n{sep}")
            output_parts.append(" 参考文献・出典リスト")
            output_parts.append(sep)
            output_parts.append(reference_list)

        output = "\n".join(output_parts)
        print(output)
        print(f"\n  全7ステップ完了！総処理時間: {total_elapsed:.1f}秒")

        if output_path is None:
            if material_path:
                stem = Path(material_path).stem
                output_path = str(Path(material_path).parent / f"{stem}_原稿.txt")
            else:
                output_path = f"原稿_{cp_id}.txt"
        Path(output_path).write_text(output, encoding="utf-8")
        print(f"\n  結果を保存しました: {output_path}")

    except Exception as e:
        # エラー時にチェックポイントにエラー情報を追記
        try:
            existing = load_checkpoint(cp_id) or {}
            existing["error"] = str(e)
            existing["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_checkpoint(cp_id, existing)
        except Exception:
            pass
        print(f"\n  ❌ エラー: {e}")
        print(f"  💾 チェックポイント保存済み: {cp_id}")
        print(f"  再開するには: python manuscript_checker.py --resume {cp_id}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="資料→原稿化ワークフローシステム v2 (7ステップ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用例:
  python manuscript_checker.py 資料.txt --title "仮タイトル" --purpose "趣旨"
  python manuscript_checker.py 資料.txt --title "タイトル" --purpose "趣旨" -o 結果.txt
  python manuscript_checker.py 資料.txt --title "タイトル" --purpose "趣旨" --no-search
  python manuscript_checker.py --resume abc12345
  python manuscript_checker.py --list-checkpoints
  python manuscript_checker.py --list-history

環境変数:
  ANTHROPIC_API_KEY  Anthropic APIキー (必須)
""",
    )
    parser.add_argument("material", nargs="?", help="資料ファイルパス")
    parser.add_argument("--title", "-t", default="", help="仮タイトル（必須）")
    parser.add_argument("--purpose", "-p", default="", help="動画の趣旨・テーマ（必須）")
    parser.add_argument("-o", "--output", help="出力ファイルパス (デフォルト: 資料名_原稿.txt)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"批評用モデル (デフォルト: {DEFAULT_MODEL})",
    )
    parser.add_argument("--no-search", action="store_true", help="Web検索を無効にする")
    parser.add_argument("--resume", metavar="CHECKPOINT_ID", help="チェックポイントから途中再開")
    parser.add_argument("--list-checkpoints", action="store_true", help="保存済みチェックポイント一覧を表示")
    parser.add_argument("--list-history", action="store_true", help="完了済みジョブの履歴一覧を表示")

    args = parser.parse_args()

    # 履歴一覧表示
    if args.list_history:
        items = list_history_items()
        if not items:
            print("完了済みジョブの履歴はありません。")
        else:
            print(f"\n📋 完了済みジョブ履歴 ({len(items)}件):\n")
            for h in items:
                print(f"  ID: {h['job_id']}  {h['completed_at']}  ({h['total_elapsed']}秒)")
                print(f"      タイトル: {h['title_preview']}")
                print(f"      趣旨: {h['purpose_preview']}")
                print(f"      資料: {h['material_preview'][:60]}...")
                print()
        return

    # チェックポイント一覧表示
    if args.list_checkpoints:
        cps = list_checkpoints()
        if not cps:
            print("保存済みチェックポイントはありません。")
        else:
            print(f"\n💾 保存済みチェックポイント ({len(cps)}件):\n")
            for cp in cps:
                step_label = f"Step {cp['completed_step']}/7 完了"
                error_label = f" [エラー: {cp['error'][:60]}]" if cp.get("error") else ""
                print(f"  ID: {cp['id']}  {step_label}  {cp['timestamp']}{error_label}")
                print(f"      タイトル: {cp['title_preview']}")
                print(f"      趣旨: {cp['purpose_preview']}")
                print(f"      資料: {cp['material_preview'][:60]}...")
                print(f"      再開: python manuscript_checker.py --resume {cp['id']}")
                print()
        return

    # 再開モード
    if args.resume:
        asyncio.run(main_async(
            None, args.title, args.purpose,
            args.output, args.model, use_search=not args.no_search,
            resume_id=args.resume,
        ))
        return

    # 通常モード
    if not args.material:
        parser.error("資料ファイルパスが必要です（--resume で再開する場合は不要）")
    if not args.title:
        parser.error("--title / -t で仮タイトルを指定してください")
    if not args.purpose:
        parser.error("--purpose / -p で趣旨を指定してください")

    asyncio.run(main_async(
        args.material, args.title, args.purpose,
        args.output, args.model, use_search=not args.no_search,
    ))


if __name__ == "__main__":
    main()
