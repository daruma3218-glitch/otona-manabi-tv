#!/usr/bin/env python3
from __future__ import annotations

"""
資料→原稿化ワークフローシステム - 3並行パイプライン版 (v3)

3本の独立パイプライン（各Step 1-6）を並列実行し、
Check Agent (Opus) が3本を比較・統合してベスト原稿を生成。
その後ファクトチェック＋参考文献リストで仕上げる。

使い方:
    export ANTHROPIC_API_KEY='your-api-key'
    python3 app_triple.py

ブラウザで http://localhost:8083 を開く
"""

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, request, jsonify

import anthropic

app = Flask(__name__)

# 進行中のジョブを管理
jobs: dict[str, dict] = {}

# ──────────────────────────────────────────────
# ストレージ設定（Supabase or ローカルファイル）
# ──────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

if SUPABASE_URL and SUPABASE_KEY:
    from supabase import create_client
    _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    USE_SUPABASE = True
    print("✅ Supabase 接続: 履歴・チェックポイントをDBに永続化")
else:
    _supabase = None
    USE_SUPABASE = False
    print("📁 ローカルファイルモード: 履歴・チェックポイントをファイルに保存")

# ローカルフォールバック用ディレクトリ
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints_triple"
CHECKPOINT_DIR.mkdir(exist_ok=True)
HISTORY_DIR = Path(__file__).parent / "history_triple"
HISTORY_DIR.mkdir(exist_ok=True)
MAX_HISTORY = 50

# パイプライン並行数
PIPELINE_COUNT = 3
PIPELINE_IDS = ["A", "B", "C"]
MAX_CONCURRENT_API_CALLS = 5

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
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
MAX_RETRIES = 12
RETRY_INITIAL_WAIT = 30
RETRY_MAX_WAIT = 180
STEP_COOLDOWN = 3
EXPERT_COOLDOWN = 2
NUM_EXPERTS = 3
PARALLEL_BATCH_SIZE = 5

# --- Step 0: 自動リサーチ（タイトル＋趣旨 → 資料テキスト生成） ---
AUTO_RESEARCH_MAX_TOKENS = 16000

AUTO_RESEARCH_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 20,
}

AUTO_RESEARCH_SYSTEM = """\
あなたはYouTubeチャンネル「大人の学び直しTV」の専属リサーチャーです。
与えられたテーマについて徹底的にWeb検索を行い、原稿作成チームに渡すための「資料テキスト」を作成してください。

あなたの出力は、この後の工程で複数の専門家エージェントと原稿作成AIが読み込む「唯一の情報源」になります。
情報の網羅性・正確性・多角性がそのまま原稿の品質を決めます。

【リサーチの原則】
1. 最低10回以上のWeb検索を行うこと（異なる切り口で）
2. 賛成論・反対論・中立的分析を必ずバランスよく収集する
3. 国民の生活実感・世論の温度感を必ず含める（制度側の論理だけにしない）
4. 数値データは具体的な数字と出典を明記する
5. 最新の情報（直近1年以内）を優先する
6. 海外の類似事例や比較データも探す

【検索の切り口（これらを網羅的に）】
- テーマの基本的な事実・背景
- 賛成派の主張と根拠
- 反対派の主張と根拠
- 政府・省庁の公式見解
- 国民の反応・世論調査・SNSの声
- 具体的な数値データ・統計
- 海外の事例・比較
- 専門家・識者の分析
- 歴史的経緯・過去の類似事例
- 今後の見通し・予測
"""

AUTO_RESEARCH_USER_PROMPT = """\
以下のテーマについて徹底的にリサーチし、資料テキストを作成してください。

【テーマ（仮タイトル）】
{tentative_title}

【趣旨】
{purpose}

━━━ リサーチ方法 ━━━
1. まずテーマの基本情報を検索
2. 見つかったキーワードや論点をもとに、さらに深掘り検索を繰り返す
3. 賛成論・反対論・国民の声など、多角的な視点で検索を重ねる
4. 最低10回以上の検索を行い、十分な情報量を確保する
━━━━━━━━━━━━━

━━━ 出力形式 ━━━
以下の構成で資料テキストを作成してください:

## 1. テーマの概要と背景
（基本的な事実関係、経緯、現状を整理）

## 2. 賛成側の主張と根拠
（推進派の論拠、期待される効果、具体的データ）

## 3. 反対側の主張と根拠
（慎重派・反対派の論拠、懸念されるリスク、具体的データ）

## 4. 数値データ・統計
（関連する具体的な数字、国際比較、時系列変化。出典URL付き）

## 5. 国民の反応・世論
（世論調査結果、SNSでの反応傾向、生活者の実感。出典付き）

## 6. 海外の事例
（類似テーマの海外事例、日本との比較）

## 7. 専門家・識者の見解
（経済学者、政治学者、ジャーナリスト等の分析。出典付き）

## 8. 今後の見通し
（今後どうなるか、注目すべきポイント）

各セクションに出典URLを付記してください。
━━━━━━━━━━━━━
"""

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

SENTIMENT_AGENT = {
    "name": "世論感覚チェッカー",
    "system_prompt": (
        "あなたは日本に暮らす普通の納税者・生活者の感覚を代弁する存在です。\n"
        "経済学の理論ではなく、「給料から引かれる税金・社会保険料を毎月見ている人間」の目線で原稿を読んでください。\n\n"
        "あなたの仕事は、原稿が制度側・官公庁側・専門家側に偏っていないかをチェックすることです。\n"
        "具体的には:\n"
        "- 増税や負担増を「仕方がない」「必要なこと」と当然視していないか\n"
        "- 減税や給付を「財源が〜」「バラマキ〜」と否定的に扱っていないか\n"
        "- 政府・省庁の言い分をそのまま「正論」として提示していないか\n"
        "- 国民が実際に感じている負担感・不満・期待がきちんと反映されているか\n\n"
        "あなたは扇動者ではありません。「国民はこう感じている」という事実を指摘する役割です。\n"
        "改善提案は「国民の実感を反映する表現の追加」に限定してください。"
    ),
}

SENTIMENT_USER_PROMPT = """\
以下のYouTube動画原稿を、日本で働き税金を払っている30〜50代の生活者の目線でチェックしてください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

{review_concept}

━━━ 出力形式（厳守）━━━

■ 立ち位置バイアス判定:
  原稿全体が寄っている方向を判定してください:
  🏛️ 制度側寄り（官公庁・専門家の論理が前面）
  ⚖️ バランス型（両方の視点が同等に提示）
  👤 生活者寄り（国民の実感が前面）

■ 制度側に偏っている箇所（最大5つ）:
  1. 「該当箇所の引用（10〜20字）」
     → 問題: なぜ制度側に偏っているか（1文）
     → 生活者の実感: 国民はこの話をどう感じるか（1文）
     → 追加すべき視点:（1-2文で具体的に）

■ 国民の実感がよく反映されている箇所（最大3つ）:
  1. 「該当箇所の引用（10〜20字）」→ なぜ良いか（1文）

■ 世論温度:
  この原稿を見た視聴者のコメント欄を予測:
  - 最も多い反応:
  - 炎上リスクのある箇所:

※論理構造、視聴維持率、表現の良し悪しには触れないでください。「立ち位置の偏り」のみ。
━━━━━━━━━━━━━━━

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
REVIEWER_AGENTS = [CRITIC_AGENT, COMMENTATOR_AGENT, ADVISOR_AGENT, SENTIMENT_AGENT, COPYWRITER_AGENT]

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
以下の原稿に対する5人のレビュアーの意見を精査し、採用すべき意見を選定してください。
※世論感覚チェッカーが「制度側寄り」と判定した場合、その指摘は優先的に採用を検討してください。視聴者は国民であり、国民の実感とズレた原稿は信頼を失います。

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
# Check Agent（統合エージェント）プロンプト
# ──────────────────────────────────────────────
CHECK_AGENT_MAX_TOKENS = 24000

CHECK_AGENT_SYSTEM = """\
あなたは最高レベルの編集者であり、3つの独立した原稿から最高の1つの原稿を作り上げる「統合編集者」です。

3つの原稿は同じ資料・同じ趣旨から独立して作成されましたが、
それぞれ異なる視点の選び方、異なる構成、異なる表現を持っています。

あなたの仕事は「ベスト・オブ・ブリード」の統合原稿を作成することです。

━━━ 評価プロセス ━━━

まず、以下の観点で3つの原稿を比較・評価してください:

1. **主張軸の明確さ** — 動画全体の主張と各章の主張が明確で一貫しているか
2. **導入の引き付け力** — 最初の1文〜導入部で視聴者を引き込めるか
3. **章構成の論理性** — 4章の流れが「起→承→転→結」として自然で説得力があるか
4. **具体例・データの質** — 使用されている事例・数値データの説得力と信頼性
5. **表現力・読みやすさ** — 読み上げ原稿として自然で聞き取りやすいか
6. **コンセプト遵守** — 視聴者の学びと理解を第一にしているか、偏りがないか
7. **文字数の充実度** — 8,000文字以上の要件を満たし、内容が薄くないか

━━━ 統合ルール ━━━

以下のルールに従って統合原稿を作成してください:

1. 3つの原稿から「最も優れた要素」を選び取り、1つの原稿に統合する
2. 統合は以下の粒度で行う:
   - 主張軸: 3つのうち最も明確で説得力のあるものをベースにする
   - 導入部: 最も引き付け力のある導入を採用する
   - 各章: 章ごとに最も優れた原稿の内容をベースにしつつ、他の原稿の優れた要素（事例、データ、表現）を加筆する
   - タイトル: 3つの原稿で提案されたタイトルから最も適切なものを選ぶ
3. 単純なコピー＆ペーストではなく、選ばれた要素が自然に融合するよう文章を整える
4. 3つすべてで共通して使われている事例・データは信頼性が高いため優先的に採用する
5. 1つの原稿にしか登場しない独自の優れた事例・データも積極的に取り入れる
6. 統合後の原稿は4章構成・ですます調・読み上げ原稿としての体裁を維持する
7. 本編4章の合計が8,000文字以上を維持する

━━━ 使用禁止語句 ━━━
以下の言葉は使用禁止です: 「アキレス腱」「時限爆弾」「不確実性」「構造的」「ジレンマ」「明確に」「複合的な」「単なる」「ただし」「とはいえ」「しかしながら」「もっとも」

━━━ 出力形式 ━━━

まず「統合評価レポート」を出力し、その後に「統合原稿」を全文出力してください。

## 統合評価レポート
### 原稿A 評価
（各観点の評価を簡潔に）
### 原稿B 評価
（各観点の評価を簡潔に）
### 原稿C 評価
（各観点の評価を簡潔に）
### 統合方針
（どの原稿のどの要素を採用するか、章ごとの方針）

## 統合原稿
（統合された最終原稿の全文 — 省略禁止）
"""

CHECK_AGENT_USER_PROMPT = """\
以下の3つの原稿を比較・評価し、最も優れた要素を統合した「ベスト・オブ・ブリード」の統合原稿を作成してください。

【仮タイトル】
{tentative_title}

【趣旨】
{purpose}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【原稿A】
{manuscript_a}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【原稿B】
{manuscript_b}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【原稿C】
{manuscript_c}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

上記3つの原稿から、統合評価レポートと統合原稿を出力してください。
統合原稿は省略せず全文を出力してください。
"""


# ──────────────────────────────────────────────
# PipelineEventQueue: イベント自動プレフィックス
# ──────────────────────────────────────────────

class PipelineEventQueue:
    """event_queueのラッパー。pipelineフィールドを自動付与し、イベントタイプをリネーム"""

    TYPE_MAP = {
        "step": "pipeline_step",
        "fields_selected": "pipeline_fields_selected",
        "expert_start": "pipeline_expert_start",
        "expert_done": "pipeline_expert_done",
        "draft_done": "pipeline_draft_done",
        "reviewer_start": "pipeline_reviewer_start",
        "reviewer_done": "pipeline_reviewer_done",
        "opinions_selected": "pipeline_opinions_selected",
        "rewrite_done": "pipeline_rewrite_done",
        "continuation": "pipeline_continuation",
    }

    def __init__(self, base_queue: queue.Queue, pipeline_id: str):
        self.base_queue = base_queue
        self.pipeline_id = pipeline_id

    def put(self, event: dict):
        prefixed = dict(event)
        prefixed["pipeline"] = self.pipeline_id
        event_type = prefixed.get("type", "")
        if event_type in self.TYPE_MAP:
            prefixed["type"] = self.TYPE_MAP[event_type]
        self.base_queue.put(prefixed)


# ──────────────────────────────────────────────
# チェックポイント管理（排他制御付き）
# ──────────────────────────────────────────────
_checkpoint_lock = threading.Lock()


def save_checkpoint(cp_id: str, data: dict) -> None:
    """チェックポイントを保存（スレッドセーフ）"""
    if USE_SUPABASE:
        try:
            _supabase.table("checkpoints").upsert({"id": cp_id, "data": data}).execute()
        except Exception as e:
            print(f"⚠️ Supabase checkpoint save error: {e}")
        return
    with _checkpoint_lock:
        cp_path = CHECKPOINT_DIR / f"{cp_id}.json"
        cp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(cp_id: str) -> dict | None:
    """チェックポイントを読み込む。存在しなければNone"""
    if USE_SUPABASE:
        try:
            res = _supabase.table("checkpoints").select("data").eq("id", cp_id).execute()
            if res.data:
                return res.data[0]["data"]
        except Exception as e:
            print(f"⚠️ Supabase checkpoint load error: {e}")
        return None
    cp_path = CHECKPOINT_DIR / f"{cp_id}.json"
    if cp_path.exists():
        return json.loads(cp_path.read_text(encoding="utf-8"))
    return None


def delete_checkpoint(cp_id: str) -> None:
    """チェックポイントを削除"""
    if USE_SUPABASE:
        try:
            _supabase.table("checkpoints").delete().eq("id", cp_id).execute()
        except Exception as e:
            print(f"⚠️ Supabase checkpoint delete error: {e}")
        return
    cp_path = CHECKPOINT_DIR / f"{cp_id}.json"
    if cp_path.exists():
        cp_path.unlink()


def list_checkpoints() -> list[dict]:
    """保存済みチェックポイント一覧を返す（新しい順、最大10件）"""
    if USE_SUPABASE:
        try:
            res = _supabase.table("checkpoints").select("id, data, created_at").order("created_at", desc=True).limit(10).execute()
            result = []
            for row in res.data:
                d = row["data"]
                result.append({
                    "id": row["id"],
                    "mode": "triple",
                    "completed_step": d.get("completed_step", 0),
                    "material_preview": d.get("material", "")[:80],
                    "title_preview": d.get("tentative_title", "")[:40],
                    "purpose_preview": d.get("purpose", "")[:60],
                    "timestamp": d.get("timestamp", ""),
                    "error": d.get("error", ""),
                })
            return result
        except Exception as e:
            print(f"⚠️ Supabase checkpoint list error: {e}")
            return []
    result = []
    for f in sorted(CHECKPOINT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "id": f.stem,
                "mode": "triple",
                "completed_step": data.get("completed_step", 0),
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
    if USE_SUPABASE:
        try:
            _supabase.table("history").upsert({"job_id": job_id, "data": data}).execute()
            # MAX_HISTORY 制限: 古いものを削除
            res = _supabase.table("history").select("job_id, created_at").order("created_at", desc=True).execute()
            if len(res.data) > MAX_HISTORY:
                old_ids = [r["job_id"] for r in res.data[MAX_HISTORY:]]
                for old_id in old_ids:
                    _supabase.table("history").delete().eq("job_id", old_id).execute()
        except Exception as e:
            print(f"⚠️ Supabase history save error: {e}")
        return
    h_path = HISTORY_DIR / f"{job_id}.json"
    h_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(files) > MAX_HISTORY:
        files[0].unlink()
        files.pop(0)


def list_history() -> list[dict]:
    """履歴一覧を返す（新しい順）"""
    if USE_SUPABASE:
        try:
            res = _supabase.table("history").select("job_id, data, created_at").order("created_at", desc=True).limit(MAX_HISTORY).execute()
            result = []
            for row in res.data:
                d = row["data"]
                result.append({
                    "job_id": row["job_id"],
                    "completed_at": d.get("completed_at", ""),
                    "title_preview": d.get("tentative_title", "")[:40],
                    "purpose_preview": d.get("purpose", "")[:60],
                    "material_preview": d.get("material", "")[:80],
                    "total_elapsed": d.get("total_elapsed", 0),
                    "mode": "triple",
                })
            return result
        except Exception as e:
            print(f"⚠️ Supabase history list error: {e}")
            return []
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
                "mode": "triple",
            })
        except (json.JSONDecodeError, OSError):
            continue
    return result[:MAX_HISTORY]


def load_history(job_id: str) -> dict | None:
    """特定の履歴データを読み込む"""
    if USE_SUPABASE:
        try:
            res = _supabase.table("history").select("data").eq("job_id", job_id).execute()
            if res.data:
                return res.data[0]["data"]
        except Exception as e:
            print(f"⚠️ Supabase history load error: {e}")
        return None
    h_path = HISTORY_DIR / f"{job_id}.json"
    if h_path.exists():
        return json.loads(h_path.read_text(encoding="utf-8"))
    return None


def format_history_as_text(data: dict) -> str:
    """履歴データをテキスト形式に整形（Step3×3並行版）"""
    sep = "=" * 70
    parts = [
        sep,
        " 資料→原稿化ワークフロー結果（Step3×3並行 + Check Agent統合版）",
        f" 作成日時: {data.get('completed_at', '')}",
        f" 仮タイトル: {data.get('tentative_title', '')}",
        f" 趣旨: {data.get('purpose', '')}",
        sep,
    ]

    # 専門分野（共通）
    fields = data.get("fields", [])
    if fields:
        parts.append("\n■ Step 1: 選定された専門分野:")
        for i, f in enumerate(fields, 1):
            parts.append(f"  {i}. {f.get('name', '')} - {f.get('role_description', '')}")

    # 3つの中間原稿（Step 3）
    drafts = data.get("drafts", {})
    if drafts:
        parts.append(f"\n{'━' * 70}")
        parts.append(" Step 3: 3つの中間原稿")
        parts.append(f"{'━' * 70}")
        for pid in PIPELINE_IDS:
            draft_text = drafts.get(pid, "")
            if draft_text:
                parts.append(f"\n  ── 原稿 {pid} ──")
                parts.append(draft_text[:500] + ("..." if len(draft_text) > 500 else ""))

    # 統合評価レポート（Step 4）
    merge_report = data.get("merge_report", "")
    if merge_report:
        parts.append(f"\n{sep}")
        parts.append(" Step 4: Check Agent 統合評価レポート")
        parts.append(sep)
        parts.append(merge_report)

    # 統合原稿（Step 4）
    merged = data.get("merged_draft", "")
    if merged:
        parts.append(f"\n{sep}")
        parts.append(" Step 4: 統合原稿（Check Agent）")
        parts.append(sep)
        parts.append(merged)

    # リライト済み原稿（Step 7）
    rewritten = data.get("rewritten_draft", "")
    if rewritten:
        parts.append(f"\n{sep}")
        parts.append(" Step 7: リライト済み原稿")
        parts.append(sep)
        parts.append(rewritten)

    # 最終原稿（Step 8）
    final = data.get("final", "")
    if final:
        parts.append(f"\n{sep}")
        parts.append(" 最終原稿 - ファクトチェック済み")
        parts.append(sep)
        parts.append(final)

    # 参考文献・出典リスト
    ref_list = data.get("reference_list", "")
    if ref_list:
        parts.append(f"\n{sep}")
        parts.append(" 参考文献・出典リスト")
        parts.append(sep)
        parts.append(ref_list)

    return "\n".join(parts)


# ──────────────────────────────────────────────
# API処理（semaphore対応）
# ──────────────────────────────────────────────

def extract_text_from_response(response) -> str:
    """レスポンスからテキストブロックのみを抽出して結合する。"""
    texts = []
    for block in response.content:
        if hasattr(block, "text"):
            texts.append(block.text)
    return "".join(texts)


def get_retry_wait(retry_num: int, error=None) -> float:
    """指数バックオフでリトライ待ち時間を計算。retry-afterヘッダがあればそれを優先。"""
    if error and hasattr(error, "response") and error.response is not None:
        headers = getattr(error.response, "headers", {})
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after) + 5, RETRY_MAX_WAIT)
            except (ValueError, TypeError):
                pass
    wait = min(RETRY_INITIAL_WAIT * (retry_num + 1), RETRY_MAX_WAIT)
    return wait


async def _do_api_call(client, semaphore: asyncio.Semaphore | None, **kwargs) -> object:
    """セマフォ付きAPI呼び出し（ストリーミングで実行し、最終レスポンスを返す）"""
    async def _call():
        collected_text = ""
        stop_reason = None
        usage = None
        content_blocks = []

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                pass  # ストリーム完了まで消費
            response = await stream.get_final_message()
        return response

    if semaphore:
        async with semaphore:
            return await asyncio.wait_for(_call(), timeout=600)
    else:
        return await asyncio.wait_for(_call(), timeout=600)


async def api_call_with_retry(client, event_queue, label: str, semaphore: asyncio.Semaphore | None = None, **kwargs) -> object:
    """リトライ付きAPI呼び出し（ストリーミングモード）。semaphoreで同時呼び出し数を制限。"""
    last_error = None
    for retry in range(MAX_RETRIES):
        try:
            response = await _do_api_call(client, semaphore, **kwargs)
            return response
        except asyncio.TimeoutError:
            last_error = Exception("API呼び出しが10分以上応答なし")
            wait_sec = min(RETRY_INITIAL_WAIT * (retry + 1), RETRY_MAX_WAIT)
            event_queue.put({
                "type": "continuation",
                "message": f"{label} API応答タイムアウト、{int(wait_sec)}秒後にリトライ... ({retry + 1}/{MAX_RETRIES})",
            })
            await asyncio.sleep(wait_sec)
            continue
        except anthropic.RateLimitError as e:
            last_error = e
            wait_sec = get_retry_wait(retry, e)
            event_queue.put({
                "type": "continuation",
                "message": f"{label} レート制限のため{int(wait_sec)}秒待機中... (リトライ {retry + 1}/{MAX_RETRIES})",
            })
            await asyncio.sleep(wait_sec)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            last_error = e
            wait_sec = min(RETRY_INITIAL_WAIT * (retry + 1), RETRY_MAX_WAIT)
            event_queue.put({
                "type": "continuation",
                "message": f"{label} APIエラー ({type(e).__name__})、{int(wait_sec)}秒後にリトライ... ({retry + 1}/{MAX_RETRIES})",
            })
            await asyncio.sleep(wait_sec)
        except anthropic.APIError as e:
            if hasattr(e, "status_code") and e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            last_error = e
            wait_sec = min(RETRY_INITIAL_WAIT * (retry + 1), RETRY_MAX_WAIT)
            event_queue.put({
                "type": "continuation",
                "message": f"{label} APIエラー、{int(wait_sec)}秒後にリトライ... ({retry + 1}/{MAX_RETRIES})",
            })
            await asyncio.sleep(wait_sec)

    raise Exception(f"{label} {MAX_RETRIES}回リトライしましたが失敗しました。最後のエラー: {last_error}")


async def call_with_continuation(client, model: str, system: str, messages: list[dict], max_tokens: int, event_queue, label: str = "", use_search: bool = False, search_tool: dict | None = None, semaphore: asyncio.Semaphore | None = None) -> str:
    """API呼び出し＋途切れた場合に自動で続きを取得。semaphore対応。"""
    full_text = ""
    current_messages = list(messages)

    tools = [search_tool or WEB_SEARCH_TOOL] if use_search else []

    for attempt in range(MAX_CONTINUATIONS + 1):
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=current_messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = await api_call_with_retry(client, event_queue, label, semaphore=semaphore, **kwargs)

        chunk = extract_text_from_response(response)
        full_text += chunk

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "pause_turn" and attempt < MAX_CONTINUATIONS:
            event_queue.put({
                "type": "continuation",
                "message": f"{label} 検索結果を処理中... ({attempt + 1}回目)",
            })
            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": "続けてください。"},
            ]
            continue

        if response.stop_reason == "max_tokens" and attempt < MAX_CONTINUATIONS:
            event_queue.put({
                "type": "continuation",
                "message": f"{label} 出力が長いため続きを取得中... ({attempt + 1}回目)",
            })
            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": "途切れてしまいました。直前の続きから、最後まで完成させてください。重複せず、途切れた箇所のすぐ続きから書いてください。"},
            ]
        else:
            break

    return full_text


# ──────────────────────────────────────────────
# Step関数（既存と同一。event_queueにPipelineEventQueueを渡すことで自動プレフィックス）
# ──────────────────────────────────────────────

# --- Step 0: 自動リサーチ ---
async def auto_research_topic(client, tentative_title: str, purpose: str, model: str, event_queue) -> str:
    """タイトルと趣旨からWeb検索で資料テキストを自動生成"""
    event_queue.put({
        "type": "step",
        "step": 0,
        "message": f"テーマを自動リサーチ中...（{model} + Web検索20回）",
    })

    user_prompt = AUTO_RESEARCH_USER_PROMPT.format(
        tentative_title=tentative_title,
        purpose=purpose,
    )
    messages = [{"role": "user", "content": user_prompt}]

    result = await call_with_continuation(
        client, model, AUTO_RESEARCH_SYSTEM, messages,
        AUTO_RESEARCH_MAX_TOKENS, event_queue,
        label="自動リサーチ",
        use_search=True,
        search_tool=AUTO_RESEARCH_SEARCH_TOOL,
    )

    event_queue.put({
        "type": "auto_research_done",
        "material": result,
    })

    return result


# --- Step 1: 分野選定 ---
async def select_expert_fields(client, material: str, model: str, event_queue, semaphore: asyncio.Semaphore | None = None) -> list[dict]:
    """資料を分析し、最適な3つの専門分野を自動選定"""
    event_queue.put({"type": "step", "step": 1, "message": "資料を分析し、最適な専門分野を選定中..."})

    response = await api_call_with_retry(
        client, event_queue, "分野選定",
        semaphore=semaphore,
        model=model,
        max_tokens=2048,
        system=FIELD_SELECTION_SYSTEM,
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

    event_queue.put({
        "type": "fields_selected",
        "fields": [{"name": f["name"], "role_description": f["role_description"]} for f in fields],
    })

    return fields


# --- Step 2: 専門家視点チェック ---
async def run_expert_review(client, material: str, field: dict, index: int, model: str, event_queue, use_search: bool = False, semaphore: asyncio.Semaphore | None = None) -> dict:
    """1人の専門家エージェントによる視点チェック"""
    event_queue.put({"type": "expert_start", "index": index, "name": field["name"]})

    start_time = time.time()

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

    messages = [{"role": "user", "content": user_prompt}]
    review_text = await call_with_continuation(
        client, model, field["system_prompt"], messages,
        EXPERT_MAX_TOKENS, event_queue,
        label=f"専門家{index + 1}({field['name']})",
        use_search=use_search,
        semaphore=semaphore,
    )

    elapsed = time.time() - start_time
    event_queue.put({
        "type": "expert_done",
        "index": index,
        "name": field["name"],
        "review": review_text,
        "elapsed": round(elapsed, 1),
    })

    return {
        "index": index,
        "field_name": field["name"],
        "role_description": field["role_description"],
        "review": review_text,
        "elapsed": elapsed,
    }


# --- Step 3: 原稿化 ---
async def draft_manuscript(client, material: str, expert_results: list[dict], tentative_title: str, purpose: str, event_queue, semaphore: asyncio.Semaphore | None = None) -> str:
    """専門家チェック結果を踏まえて原稿を作成（Opus 4.6）"""
    event_queue.put({"type": "step", "step": 3, "message": f"4章構成の原稿を作成中...（{INTEGRATION_MODEL}）"})

    expert_reviews_text = ""
    for r in expert_results:
        expert_reviews_text += f"\n{'─' * 50}\n"
        expert_reviews_text += f"【専門家{r['index'] + 1}: {r['field_name']}の視点チェック】\n"
        expert_reviews_text += f"（役割: {r['role_description']}）\n\n"
        expert_reviews_text += r["review"]
        expert_reviews_text += "\n"

    messages = [{
        "role": "user",
        "content": DRAFTING_USER_PROMPT.format(
            material=material,
            expert_reviews=expert_reviews_text,
            tentative_title=tentative_title,
            purpose=purpose,
        ),
    }]

    draft = await call_with_continuation(
        client, INTEGRATION_MODEL, DRAFTING_SYSTEM, messages,
        DRAFTING_MAX_TOKENS, event_queue, label="原稿化エージェント(Opus)",
        semaphore=semaphore,
    )

    event_queue.put({
        "type": "draft_done",
        "draft": draft,
    })

    return draft


# --- Step 4: 批評フェーズ ---
async def run_single_reviewer(client, draft: str, agent: dict, index: int, tentative_title: str, purpose: str, model: str, event_queue, semaphore: asyncio.Semaphore | None = None) -> dict:
    """1人のレビュアーによる批評"""
    event_queue.put({"type": "reviewer_start", "index": index, "name": agent["name"]})

    start_time = time.time()

    # エージェントごとの専用プロンプトを選択
    prompt_map = {
        "批評家": CRITIC_USER_PROMPT,
        "評論家（補足説明）": COMMENTATOR_USER_PROMPT,
        "アドバイザー": ADVISOR_USER_PROMPT,
        "世論感覚チェッカー": SENTIMENT_USER_PROMPT,
        "コピーライター": COPYWRITER_USER_PROMPT,
    }
    user_prompt_template = prompt_map.get(agent["name"], CRITIC_USER_PROMPT)
    user_prompt = user_prompt_template.format(
        draft=draft,
        tentative_title=tentative_title,
        purpose=purpose,
        review_concept=REVIEW_CONCEPT,
    )

    messages = [{"role": "user", "content": user_prompt}]
    review_text = await call_with_continuation(
        client, model, agent["system_prompt"], messages,
        REVIEWER_MAX_TOKENS, event_queue,
        label=f"レビュアー{index + 1}({agent['name']})",
        semaphore=semaphore,
    )

    elapsed = time.time() - start_time
    event_queue.put({
        "type": "reviewer_done",
        "index": index,
        "name": agent["name"],
        "review": review_text,
        "elapsed": round(elapsed, 1),
    })

    return {
        "index": index,
        "name": agent["name"],
        "review": review_text,
        "elapsed": elapsed,
    }


async def run_review_phase(client, draft: str, tentative_title: str, purpose: str, model: str, event_queue, semaphore: asyncio.Semaphore | None = None) -> list[dict]:
    """4人のレビュアーを並列実行"""
    event_queue.put({"type": "step", "step": 4, "message": "4人のレビュアーが原稿を批評中..."})

    tasks = [
        run_single_reviewer(client, draft, agent, i, tentative_title, purpose, model, event_queue, semaphore=semaphore)
        for i, agent in enumerate(REVIEWER_AGENTS)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    review_results = []
    for r in results:
        if isinstance(r, Exception):
            raise r
        review_results.append(r)

    return sorted(review_results, key=lambda r: r["index"])


# --- Step 5: 意見選定 ---
async def select_opinions(client, draft: str, review_results: list[dict], tentative_title: str, purpose: str, event_queue, semaphore: asyncio.Semaphore | None = None) -> str:
    """コンセプトに沿った意見を取捨選択 + タイトル選定"""
    event_queue.put({"type": "step", "step": 5, "message": f"意見を取捨選択中...（{INTEGRATION_MODEL}）"})

    review_text = ""
    for r in review_results:
        review_text += f"\n{'━' * 50}\n"
        review_text += f"【{r['name']}の意見】\n\n"
        review_text += r["review"]
        review_text += "\n"

    messages = [{
        "role": "user",
        "content": OPINION_SELECTION_USER_PROMPT.format(
            draft=draft,
            review_results=review_text,
            tentative_title=tentative_title,
            purpose=purpose,
        ),
    }]

    result = await call_with_continuation(
        client, INTEGRATION_MODEL, OPINION_SELECTION_SYSTEM, messages,
        OPINION_MAX_TOKENS, event_queue, label="意見選定(Opus)",
        semaphore=semaphore,
    )

    event_queue.put({
        "type": "opinions_selected",
        "opinions": result,
    })

    return result


# --- Step 6: リライト ---
async def rewrite_manuscript(client, draft: str, selected_opinions: str, event_queue, semaphore: asyncio.Semaphore | None = None) -> str:
    """選定意見を反映してリライト"""
    event_queue.put({"type": "step", "step": 6, "message": f"リライト中...（{INTEGRATION_MODEL}）"})

    messages = [{
        "role": "user",
        "content": REWRITE_USER_PROMPT.format(
            draft=draft,
            selected_opinions=selected_opinions,
        ),
    }]

    result = await call_with_continuation(
        client, INTEGRATION_MODEL, REWRITE_SYSTEM, messages,
        REWRITE_MAX_TOKENS, event_queue, label="リライト(Opus)",
        semaphore=semaphore,
    )

    event_queue.put({
        "type": "rewrite_done",
        "rewritten": result,
    })

    return result


# --- ファクトチェック ---
async def fact_check_draft(client, draft: str, event_queue, semaphore: asyncio.Semaphore | None = None) -> tuple:
    """原稿をWeb検索でファクトチェックし、修正版原稿を返す"""
    event_queue.put({"type": "step", "step": 8, "message": f"ファクトチェック中...（{DEFAULT_MODEL} + Web検索）"})

    messages = [{
        "role": "user",
        "content": FACT_CHECK_USER_PROMPT.format(draft=draft),
    }]

    result = await call_with_continuation(
        client, DEFAULT_MODEL, FACT_CHECK_SYSTEM, messages,
        FACT_CHECK_MAX_TOKENS, event_queue, label="ファクトチェック(Sonnet)",
        use_search=True, search_tool=FACT_CHECK_SEARCH_TOOL,
        semaphore=semaphore,
    )

    # ファクトチェック結果と修正版原稿を分離
    fact_report = ""
    corrected_draft = result

    split_markers = ["## 修正版原稿", "# 修正版原稿", "##修正版原稿", "#修正版原稿",
                     "## 修正済み原稿", "# 修正済み原稿", "## 修正後原稿", "## ファクトチェック修正版"]
    found = False
    for marker in split_markers:
        if marker in result:
            parts = result.split(marker, 1)
            fact_report = parts[0].strip()
            corrected_draft = parts[1].strip()
            found = True
            break

    if not found and result:
        event_queue.put({
            "type": "continuation",
            "message": "ファクトチェック: 修正版原稿の区切りが見つかりませんでした。全文を修正版として使用します。",
        })

    event_queue.put({
        "type": "fact_check_done",
        "fact_report": fact_report if fact_report else "（ファクトチェック結果は修正版原稿に統合済み）",
    })

    return corrected_draft, fact_report


async def generate_reference_list(
    client, final_draft: str, fact_report: str,
    all_expert_results: list[dict], event_queue,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """最終原稿と情報源データから参考文献・出典リストを生成する"""
    event_queue.put({
        "type": "step",
        "step": "8b",
        "message": "参考文献・出典リストを生成中...",
    })

    # 全パイプラインの専門家リサーチ結果からURL含む情報を組み立て
    expert_sources_text = ""
    for r in all_expert_results:
        expert_sources_text += f"\n--- {r.get('field_name', '')} ---\n"
        expert_sources_text += r.get("review", "")
        expert_sources_text += "\n"

    messages = [{
        "role": "user",
        "content": REFERENCE_LIST_USER_PROMPT.format(
            final_draft=final_draft,
            fact_report=fact_report if fact_report else "（ファクトチェック結果なし）",
            expert_sources=expert_sources_text,
        ),
    }]

    result = await call_with_continuation(
        client, DEFAULT_MODEL, REFERENCE_LIST_SYSTEM, messages,
        REFERENCE_LIST_MAX_TOKENS, event_queue, label="参考文献リスト生成",
        use_search=True, search_tool=REFERENCE_LIST_SEARCH_TOOL,
        semaphore=semaphore,
    )

    event_queue.put({
        "type": "reference_list_done",
        "reference_list": result,
    })

    return result


# ──────────────────────────────────────────────
# Step 3 を3回並行実行する原稿化関数
# ──────────────────────────────────────────────

async def draft_manuscript_labeled(
    pipeline_id: str,
    client, material: str, expert_results: list[dict],
    tentative_title: str, purpose: str,
    base_event_queue: queue.Queue,
    semaphore: asyncio.Semaphore,
) -> dict:
    """1つの原稿化を実行し、結果dictを返す。PipelineEventQueueで自動ID付与。"""
    event_queue = PipelineEventQueue(base_event_queue, pipeline_id)
    start_time = time.time()

    event_queue.put({"type": "step", "step": 3, "message": f"原稿{pipeline_id}を作成中...（{INTEGRATION_MODEL}）"})

    expert_reviews_text = ""
    for r in expert_results:
        expert_reviews_text += f"\n{'─' * 50}\n"
        expert_reviews_text += f"【専門家{r['index'] + 1}: {r['field_name']}の視点チェック】\n"
        expert_reviews_text += f"（役割: {r['role_description']}）\n\n"
        expert_reviews_text += r["review"]
        expert_reviews_text += "\n"

    messages = [{
        "role": "user",
        "content": DRAFTING_USER_PROMPT.format(
            material=material,
            expert_reviews=expert_reviews_text,
            tentative_title=tentative_title,
            purpose=purpose,
        ),
    }]

    draft = await call_with_continuation(
        client, INTEGRATION_MODEL, DRAFTING_SYSTEM, messages,
        DRAFTING_MAX_TOKENS, event_queue,
        label=f"原稿化{pipeline_id}(Opus)",
        semaphore=semaphore,
    )

    elapsed = time.time() - start_time
    event_queue.put({
        "type": "draft_done",
        "draft": draft,
    })
    base_event_queue.put({
        "type": "pipeline_complete",
        "pipeline": pipeline_id,
        "elapsed": round(elapsed, 1),
    })

    return {"pipeline_id": pipeline_id, "draft": draft, "elapsed": round(elapsed, 1)}


# ──────────────────────────────────────────────
# Check Agent（統合フェーズ）
# ──────────────────────────────────────────────

async def check_agent_merge(
    client,
    manuscripts: dict,
    tentative_title: str,
    purpose: str,
    event_queue: queue.Queue,
    semaphore: asyncio.Semaphore,
) -> tuple:
    """3つ（または2つ）の中間原稿を比較・統合し、ベスト原稿を生成する。

    Returns:
        (merge_report, merged_draft) のタプル
    """
    event_queue.put({
        "type": "merge_step",
        "step": 4,
        "message": f"Check Agent: {len(manuscripts)}つの中間原稿を比較・統合中...（{INTEGRATION_MODEL}）",
    })

    # 原稿をA/B/Cの順にマッピング
    manuscript_a = manuscripts.get("A", "（原稿A: 失敗）")
    manuscript_b = manuscripts.get("B", "（原稿B: 失敗）")
    manuscript_c = manuscripts.get("C", "（原稿C: 失敗）")

    user_content = CHECK_AGENT_USER_PROMPT.format(
        tentative_title=tentative_title,
        purpose=purpose,
        manuscript_a=manuscript_a,
        manuscript_b=manuscript_b,
        manuscript_c=manuscript_c,
    )

    messages = [{"role": "user", "content": user_content}]

    result = await call_with_continuation(
        client, INTEGRATION_MODEL, CHECK_AGENT_SYSTEM, messages,
        CHECK_AGENT_MAX_TOKENS, event_queue,
        label="Check Agent(Opus)",
        semaphore=semaphore,
    )

    # 統合評価レポートと統合原稿を分離
    merge_report = ""
    merged_draft = result

    split_markers = ["## 統合原稿", "# 統合原稿", "##統合原稿", "#統合原稿"]
    for marker in split_markers:
        if marker in result:
            parts = result.split(marker, 1)
            merge_report = parts[0].strip()
            merged_draft = parts[1].strip()
            break

    event_queue.put({
        "type": "merge_done",
        "merge_report": merge_report,
        "merged": merged_draft,
    })

    return merge_report, merged_draft


# ──────────────────────────────────────────────
# メインパイプライン（案2: Step3だけ3並行）
# ──────────────────────────────────────────────
#
#  Step 1: 分野選定（1回）
#  Step 2: 情報収集（1回）
#  Step 3: 原稿化 ×3 並行（A/B/C）
#  Step 4: Check Agent 統合 → ベスト1本
#  Step 5: 批評（1回）
#  Step 6: 意見選定（1回）
#  Step 7: リライト（1回）
#  Step 8: ファクトチェック + 参考文献
# ──────────────────────────────────────────────

async def run_triple_pipeline(
    material: str,
    tentative_title: str,
    purpose: str,
    model: str,
    api_key: str,
    use_search: bool,
    event_queue: queue.Queue,
    checkpoint_id: str | None = None,
    auto_research: bool = False,
):
    """Step3（原稿化）を3並行 → Check Agent統合 → 批評〜ファクトチェック"""
    cp_id = checkpoint_id or str(uuid.uuid4())[:8]
    cp_data = load_checkpoint(cp_id) if checkpoint_id else None
    start_step = (cp_data.get("completed_step", 0) + 1) if cp_data else 1

    # チェックポイントから中間データ復元
    fields = cp_data.get("fields") if cp_data else None
    expert_results = cp_data.get("expert_results") if cp_data else None
    drafts = cp_data.get("drafts") if cp_data else None  # {"A": "...", "B": "...", "C": "..."}
    merge_report = cp_data.get("merge_report") if cp_data else None
    merged_draft = cp_data.get("merged_draft") if cp_data else None
    review_results = cp_data.get("review_results") if cp_data else None
    selected_opinions = cp_data.get("selected_opinions") if cp_data else None
    rewritten_draft = cp_data.get("rewritten_draft") if cp_data else None

    def cp_common():
        return {
            "mode": "triple",
            "material": material, "tentative_title": tentative_title,
            "purpose": purpose,
            "model": model, "use_search": use_search,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
        total_start = time.time()

        if start_step > 1:
            event_queue.put({
                "type": "continuation",
                "message": f"チェックポイントからStep {start_step}より再開します",
            })

        # ──── Step 0: 自動リサーチ（auto_research=Trueかつ資料が空の場合） ────
        if auto_research and not material:
            material = await auto_research_topic(client, tentative_title, purpose, model, event_queue)
            await asyncio.sleep(STEP_COOLDOWN)

        # ──── Step 1: 分野選定（1回） ────
        if start_step <= 1:
            fields = await select_expert_fields(client, material, model, event_queue, semaphore=semaphore)
            save_checkpoint(cp_id, {
                "completed_step": 1, "fields": fields, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # ──── Step 2: 情報収集（1回） ────
        if start_step <= 2:
            num_total_experts = len(fields)
            search_label = "（Web検索あり）" if use_search else ""
            event_queue.put({"type": "step", "step": 2, "message": f"{num_total_experts}人の専門家が資料をリサーチ中...{search_label}"})
            expert_results = []
            for batch_start in range(0, num_total_experts, PARALLEL_BATCH_SIZE):
                if batch_start > 0:
                    await asyncio.sleep(EXPERT_COOLDOWN)
                batch = fields[batch_start:batch_start + PARALLEL_BATCH_SIZE]
                tasks = [
                    run_expert_review(client, material, field, batch_start + j, model, event_queue, use_search=use_search, semaphore=semaphore)
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

        # ──── Step 3: 原稿化 ×3 並行 ────
        if start_step <= 3:
            event_queue.put({
                "type": "step", "step": 3,
                "message": f"3つの中間原稿を並行作成中...（{INTEGRATION_MODEL} ×3）",
            })

            draft_tasks = [
                draft_manuscript_labeled(pid, client, material, expert_results, tentative_title, purpose, event_queue, semaphore)
                for pid in PIPELINE_IDS
            ]
            draft_results = await asyncio.gather(*draft_tasks, return_exceptions=True)

            # 2-of-3ポリシー
            drafts = {}
            failed = []
            for i, r in enumerate(draft_results):
                pid = PIPELINE_IDS[i]
                if isinstance(r, Exception):
                    failed.append(pid)
                    event_queue.put({
                        "type": "pipeline_error", "pipeline": pid,
                        "message": f"原稿{pid}がエラー: {r}",
                    })
                else:
                    drafts[pid] = r["draft"]

            if len(drafts) < 2:
                raise Exception(f"3本中{len(failed)}本の原稿化が失敗しました（最低2本必要）。失敗: {failed}")

            if failed:
                event_queue.put({
                    "type": "continuation",
                    "message": f"原稿{','.join(failed)}が失敗しましたが、残り{len(drafts)}本で統合を続行します。",
                })

            save_checkpoint(cp_id, {
                "completed_step": 3, "fields": fields, "expert_results": expert_results,
                "drafts": drafts, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # ──── Step 4: Check Agent 統合 ────
        if start_step <= 4:
            merge_report, merged_draft = await check_agent_merge(
                client, drafts, tentative_title, purpose, event_queue, semaphore,
            )
            save_checkpoint(cp_id, {
                "completed_step": 4, "fields": fields, "expert_results": expert_results,
                "drafts": drafts, "merge_report": merge_report, "merged_draft": merged_draft,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # ──── Step 5: 批評（統合原稿に対して1回） ────
        if start_step <= 5:
            review_results = await run_review_phase(client, merged_draft, tentative_title, purpose, model, event_queue, semaphore=semaphore)
            save_checkpoint(cp_id, {
                "completed_step": 5, "fields": fields, "expert_results": expert_results,
                "drafts": drafts, "merge_report": merge_report, "merged_draft": merged_draft,
                "review_results": review_results, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # ──── Step 6: 意見選定 ────
        if start_step <= 6:
            selected_opinions = await select_opinions(client, merged_draft, review_results, tentative_title, purpose, event_queue, semaphore=semaphore)
            save_checkpoint(cp_id, {
                "completed_step": 6, "fields": fields, "expert_results": expert_results,
                "drafts": drafts, "merge_report": merge_report, "merged_draft": merged_draft,
                "review_results": review_results, "selected_opinions": selected_opinions,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # ──── Step 7: リライト ────
        if start_step <= 7:
            rewritten_draft = await rewrite_manuscript(client, merged_draft, selected_opinions, event_queue, semaphore=semaphore)
            save_checkpoint(cp_id, {
                "completed_step": 7, "fields": fields, "expert_results": expert_results,
                "drafts": drafts, "merge_report": merge_report, "merged_draft": merged_draft,
                "review_results": review_results, "selected_opinions": selected_opinions,
                "rewritten_draft": rewritten_draft, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # ──── Step 8: ファクトチェック ────
        if start_step <= 8:
            final, fact_report = await fact_check_draft(client, rewritten_draft, event_queue, semaphore=semaphore)

        # ──── Step 8b: 参考文献リスト ────
        reference_list = ""
        try:
            reference_list = await generate_reference_list(
                client, final, fact_report, expert_results, event_queue,
                semaphore=semaphore,
            )
        except Exception as ref_err:
            reference_list = f"（参考文献リストの生成に失敗しました: {ref_err}）"
            event_queue.put({
                "type": "continuation",
                "message": f"参考文献リスト生成エラー: {ref_err}（原稿自体は正常に完成しています）",
            })

        total_elapsed = time.time() - total_start

        # 履歴に保存
        save_history(cp_id, {
            "mode": "triple",
            "material": material,
            "tentative_title": tentative_title,
            "purpose": purpose,
            "fields": fields,
            "expert_results": expert_results,
            "drafts": drafts,
            "merge_report": merge_report,
            "merged_draft": merged_draft,
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
        delete_checkpoint(cp_id)

        event_queue.put({
            "type": "complete",
            "final": final,
            "merged_draft": merged_draft,
            "merge_report": merge_report,
            "reference_list": reference_list,
            "total_elapsed": round(total_elapsed, 1),
        })

    except Exception as e:
        try:
            existing = load_checkpoint(cp_id) or {}
            existing["error"] = str(e)
            existing["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_checkpoint(cp_id, existing)
        except Exception:
            pass
        event_queue.put({"type": "error", "message": str(e), "checkpoint_id": cp_id})


def run_in_thread(material: str, tentative_title: str, purpose: str, model: str, api_key: str, use_search: bool, event_queue: queue.Queue, checkpoint_id: str | None = None, auto_research: bool = False):
    """別スレッドでasyncioイベントループを実行"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_triple_pipeline(material, tentative_title, purpose, model, api_key, use_search, event_queue, checkpoint_id, auto_research=auto_research))
    loop.close()


# ──────────────────────────────────────────────
# Flaskルート
# ──────────────────────────────────────────────

@app.route("/")
def index():
    import pathlib
    html_path = pathlib.Path(__file__).parent / "templates" / "index_triple.html"
    return html_path.read_text(encoding="utf-8")


@app.route("/api/start", methods=["POST"])
def start_job():
    data = request.get_json()
    material = data.get("material", "").strip()
    tentative_title = data.get("tentative_title", "").strip()
    purpose = data.get("purpose", "").strip()
    model = data.get("model", DEFAULT_MODEL)
    api_key = data.get("api_key", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    use_search = data.get("use_search", True)

    auto_research = data.get("auto_research", False)

    if not material and not auto_research:
        return jsonify({"error": "資料が空です（自動リサーチを使うか、資料を入力してください）"}), 400
    if not tentative_title:
        return jsonify({"error": "仮タイトルを入力してください"}), 400
    if not purpose:
        return jsonify({"error": "趣旨を入力してください"}), 400
    if not api_key:
        return jsonify({"error": "APIキーが設定されていません"}), 400

    job_id = str(uuid.uuid4())[:8]
    event_queue = queue.Queue()
    jobs[job_id] = {"queue": event_queue, "status": "running"}

    thread = threading.Thread(
        target=run_in_thread,
        args=(material, tentative_title, purpose, model, api_key, use_search, event_queue),
        kwargs={"auto_research": auto_research},
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/resume", methods=["POST"])
def resume_job():
    """チェックポイントから途中再開"""
    data = request.get_json()
    checkpoint_id = data.get("checkpoint_id", "").strip()
    api_key = data.get("api_key", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "")

    if not checkpoint_id:
        return jsonify({"error": "チェックポイントIDが必要です"}), 400
    if not api_key:
        return jsonify({"error": "APIキーが設定されていません"}), 400

    cp_data = load_checkpoint(checkpoint_id)
    if not cp_data:
        return jsonify({"error": f"チェックポイント {checkpoint_id} が見つかりません"}), 404

    material = cp_data.get("material", "")
    tentative_title = cp_data.get("tentative_title", "")
    purpose = cp_data.get("purpose", "")

    if not material:
        return jsonify({"error": "チェックポイントに資料データがありません。新規で開始してください。"}), 400
    model = cp_data.get("model", DEFAULT_MODEL)
    use_search = cp_data.get("use_search", True)

    job_id = str(uuid.uuid4())[:8]
    event_queue = queue.Queue()
    jobs[job_id] = {"queue": event_queue, "status": "running"}

    thread = threading.Thread(
        target=run_in_thread,
        args=(material, tentative_title, purpose, model, api_key, use_search, event_queue, checkpoint_id),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "resume_from_step": cp_data.get("completed_step", 0) + 1})


@app.route("/api/health")
def health_check():
    """Supabase接続診断"""
    info = {"supabase": USE_SUPABASE, "url_set": bool(SUPABASE_URL), "key_set": bool(SUPABASE_KEY)}
    if USE_SUPABASE:
        try:
            res = _supabase.table("history").select("job_id", count="exact").limit(1).execute()
            info["db_connected"] = True
            info["history_count"] = res.count if res.count is not None else len(res.data)
            res2 = _supabase.table("checkpoints").select("id", count="exact").limit(1).execute()
            info["checkpoints_count"] = res2.count if res2.count is not None else len(res2.data)
        except Exception as e:
            info["db_connected"] = False
            info["error"] = str(e)
    return jsonify(info)


@app.route("/api/checkpoints")
def get_checkpoints():
    """保存済みチェックポイント一覧を返す"""
    return jsonify({"checkpoints": list_checkpoints()})


@app.route("/api/history")
def get_history():
    """履歴一覧を返す"""
    return jsonify({"history": list_history()})


@app.route("/api/history/<job_id>")
def get_history_detail(job_id):
    """特定の履歴詳細を返す"""
    data = load_history(job_id)
    if not data:
        return jsonify({"error": "履歴が見つかりません"}), 404
    return jsonify(data)


@app.route("/api/download/<job_id>")
def download_history(job_id):
    """履歴をテキストファイルとしてダウンロード"""
    data = load_history(job_id)
    if not data:
        return jsonify({"error": "履歴が見つかりません"}), 404

    text = format_history_as_text(data)
    timestamp = data.get("completed_at", "").replace(" ", "_").replace(":", "")
    filename = f"原稿_triple_{timestamp}_{job_id}.txt"
    encoded_filename = quote(filename)

    return Response(
        text,
        mimetype="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename=\"{job_id}.txt\"; filename*=UTF-8''{encoded_filename}",
        },
    )


@app.route("/api/history/<job_id>", methods=["DELETE"])
def delete_history(job_id):
    """履歴を削除"""
    if USE_SUPABASE:
        try:
            res = _supabase.table("history").delete().eq("job_id", job_id).execute()
            if not res.data:
                return jsonify({"error": "履歴が見つかりません"}), 404
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    h_path = HISTORY_DIR / f"{job_id}.json"
    if not h_path.exists():
        return jsonify({"error": "履歴が見つかりません"}), 404
    h_path.unlink()
    return jsonify({"ok": True})


@app.route("/api/stream/<job_id>")
def stream(job_id):
    if job_id not in jobs:
        return jsonify({"error": "ジョブが見つかりません"}), 404

    def generate():
        q = jobs[job_id]["queue"]
        while True:
            try:
                event = q.get(timeout=600)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ──────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8083))
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  資料→原稿化ワークフロー v3 (Step3×3並行 + Check Agent 統合)      ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ブラウザで開く: http://localhost:{port}                            ║
║                                                                   ║
║  Step 1-2: 分野選定・情報収集（1回）                              ║
║  Step 3: 原稿化 ×3 並行（A / B / C）                             ║
║  Step 4: Check Agent 統合 → ベスト1本                            ║
║  Step 5-8: 批評 → 意見選定 → リライト → ファクトチェック          ║
║                                                                   ║
║  停止: Ctrl+C                                                     ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=port, debug=False)
