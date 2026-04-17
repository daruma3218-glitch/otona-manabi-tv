#!/usr/bin/env python3
from __future__ import annotations

"""
資料→原稿化ワークフローシステム - Webアプリ版 (v2)

7ステップ:
  Step 1: 分野選定
  Step 2: 情報収集（自動選定3人 + 指標のプロ + 具体論のプロ）
  Step 3: 原稿化 (Opus 4.6) — 4章構成YouTube原稿
  Step 4: 批評フェーズ（批評家 + 評論家 + アドバイザー + コピーライター 並列）
  Step 5: 意見選定 (Opus 4.6) — コンセプトに沿う意見を取捨選択 + タイトル選定
  Step 6: リライト (Opus 4.6) — 選定意見反映
  Step 7: ファクトチェック (Sonnet + Web検索)

使い方:
    export ANTHROPIC_API_KEY='your-api-key'
    python3 app.py

ブラウザで http://localhost:8080 を開く
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

import re

import anthropic

app = Flask(__name__)

# 進行中のジョブを管理
jobs: dict[str, dict] = {}

# チェックポイント保存ディレクトリ
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# 履歴保存ディレクトリ
HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(exist_ok=True)
MAX_HISTORY = 50

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
DEFAULT_MODEL = "claude-sonnet-4-6"
INTEGRATION_MODEL = "claude-opus-4-7"
EXPERT_MAX_TOKENS = 12000      # ディープリサーチ: 情報量拡大
REVIEWER_MAX_TOKENS = 6000     # 批評家も簡潔に
OPINION_MAX_TOKENS = 8000      # 意見選定
DRAFTING_MAX_TOKENS = 16000
FACT_CHECK_MAX_TOKENS = 12000
REWRITE_MAX_TOKENS = 16000
REFERENCE_LIST_MAX_TOKENS = 8000
MAX_CONTINUATIONS = 5
MAX_RETRIES = 6
RETRY_INITIAL_WAIT = 15
RETRY_MAX_WAIT = 90
STEP_COOLDOWN = 3
EXPERT_COOLDOWN = 2
NUM_EXPERTS = 3
PARALLEL_BATCH_SIZE = 5

# ──────────────────────────────────────────────
# チャンネルアイデンティティ（全エージェント共通）
# ──────────────────────────────────────────────
CHANNEL_IDENTITY = """\
━━━ チャンネル情報 ━━━
チャンネル名: 大人の学び直しTV
パーソナリティ: すあし社長（知的で落ち着いた語り口、上から目線にならない）
ターゲット視聴者: 30〜50代のビジネスパーソン。知的好奇心が旺盛で、仕事や人生に活かせる教養を求めている。専門家ではないが、ニュースや経済の基礎知識はある。
コンテンツ価値: 偏りのない視点で事実を整理し、視聴者自身が考え判断できる材料を提供する。扇動しない。ただし「偏りがない」とは官公庁・専門家の立場を自動的に中立とみなすことではない。国民・生活者の実感や世論感覚も同等の重みで扱い、制度側の論理だけでなく「それが国民の暮らしにどう影響するか」を常に含める。
トーン: 知的だが親しみやすい。講義ではなく「賢い友人との会話」。
━━━━━━━━━━━━━"""

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
    "max_uses": 8,
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
DRAFTING_SYSTEM = f"""\
{CHANNEL_IDENTITY}

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
■ 主張軸の設計（最初に必ず行う）
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 【最重要ルール — 違反は不可】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 必ず4章に分ける。5章以上は禁止。各章の内容は主張軸に忠実に。
2. 本編の文字数は日本語で8,000文字以上（トークン数ではなく文字数）。各章2,000文字以上を目安に。
3. 導入部分（300文字程度）に以下の固定テキストを必ず含める（変更禁止）:
   「そこで今回は、●●について解説します。最後までご覧いただければあなたの▲▲リテラシーまであがること間違いありません。 このチャンネルではそういった情報を配信しております。少しでも必要な情報だと感じたらどこのチャンネルかわからなくなる前に、ぜひチャンネル登録をよろしくお願いいたします。それでは早速やっていきましょう。」
4. 各章は以下の一方向の流れで構成する:
   ① 主張を提示する → ② 根拠・具体例・データで裏付ける → ③ 必要なら1回だけ別視点 → ④ 主張を強化して締める
   ※②と③を行き来しない。話は常に前に進める。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 【表現ルール】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

・ですます調。読み上げ原稿なので「平易な言葉遣い」を最優先。迷ったら常に平易な方を選ぶ。幼稚にはならない程度に。
・概念や流れを先に説明し、その後に固有名詞や具体例を出す。
・専門用語を使ったら直後に「●●とは〜」と補足。
・「実際に〜」「例えば〜」「●●によると〜」で事例や数値を盛り込む。
・過去の出来事でも「でした」「いました」を多用せず、「です」「ます」で没入感を維持。
・文頭に前進型接続詞を使用:「そこで」「そのため」「こうした背景から」「だからこそ」「そんな中」「ここで注目したいのは」「さらに」など。
・逆接表現は1章につき最大1回。使った直後に必ず主張を再強化する。
・すべての文末で改行し、行間を一行あける。
・一人称は「私」、二人称は「あなた」。
・登場人物の思いや考えは「」で表現して理解しやすく。
・各章のタイトルは12文字以内。「第1章」等の表記は不要。
・章内の小見出しなし。箇条書き禁止。演出指示不要。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 【禁止事項】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

・使用禁止ワード（別の表現に書き換えること）:
  「アキレス腱」「時限爆弾」「不確実性」「構造的」「ジレンマ」「明確に」「複合的な」「単なる」「ただし」「とはいえ」「しかしながら」「もっとも」「つまり一言で言えば」
・大袈裟な比喩は使わない。
・同じ接続表現を2段落連続で使わない（「つまり」で始まる段落が連続する等）。
"""

# --- Step 4: 批評フェーズ（4人のレビュアー） ---

REVIEW_CONCEPT = f"""\
{CHANNEL_IDENTITY}

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
    "name": "論証審査官",
    "system_prompt": (
        "あなたは論証分析の専門家です。原稿の各章について「主張→根拠→結論」の論証チェーンが成立しているかだけを検証してください。\n"
        "表現の良し悪しや視聴者体験には一切触れず、論理構造のみを審査します。\n"
        "改善提案をする場合は「どの根拠が足りないか」「どの飛躍を埋めるべきか」を具体的に示してください。"
    ),
}

CRITIC_USER_PROMPT = """\
以下のYouTube動画原稿の論証構造を審査してください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

{review_concept}

━━━ 出力形式（厳守）━━━
各章ごとに以下の形式で出力してください:

■ 章タイトル: 〇〇
■ 主張: （この章の主張を1文で要約）
■ 根拠: （主張を支える根拠・データ・事例を列挙）
■ 論証成立度: ◎完全 / ○概ね成立 / △飛躍あり / ✕破綻
■ 問題箇所:（△/✕の場合のみ、具体的に何が足りないか1-2文で）

※○や◎の章については問題箇所の記載は不要です。
※表現の良し悪し、退屈かどうか、視聴者体験には触れないでください。論理構造のみ。
━━━━━━━━━━━━━━━

【原稿】
{draft}
"""

COMMENTATOR_AGENT = {
    "name": "初見視聴者",
    "system_prompt": (
        "あなたは、このテーマについて予備知識がほとんどない30代の会社員です。\n"
        "YouTubeでたまたまこの動画をクリックした視聴者として、原稿を上から順に読み進めてください。\n"
        "論理構造の分析や改善提案は一切不要です。\n"
        "あなたがやることは「読みながら感じたこと」をリアルタイムで記録することだけです。"
    ),
}

COMMENTATOR_USER_PROMPT = """\
以下のYouTube動画の原稿を、初めてこのテーマに触れる30代会社員として上から順に読み進めてください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

{review_concept}

━━━ 出力形式（厳守）━━━
原稿を読み進めた順に、以下の3種類のリアクションを記録してください:

🤔 「該当箇所の引用（10〜20字）」→ なぜつまずいたか（1文）
😴 「該当箇所の引用（10〜20字）」→ なぜ退屈に感じたか（1文）
💡 「該当箇所の引用（10〜20字）」→ なぜ良いと思ったか（1文）

最後に:
■ 最も離脱リスクが高い箇所 TOP3（引用+理由1文ずつ）
■ 最も引き込まれた箇所 TOP3（引用+理由1文ずつ）

※改善提案は書かないでください。「読者としてどう感じたか」の記録のみです。
━━━━━━━━━━━━━━━

【原稿】
{draft}
"""

ADVISOR_AGENT = {
    "name": "視聴維持率コンサルタント",
    "system_prompt": (
        "あなたはYouTube動画の視聴維持率を専門とするコンサルタントです。\n"
        "過去数千本の教育系YouTube動画のアナリティクスを分析してきた経験があります。\n"
        "原稿の内容の良し悪しではなく、「動画としてどこで視聴者が離脱するか」だけを予測し、\n"
        "離脱を防ぐための具体的な改善策を提案してください。"
    ),
}

ADVISOR_USER_PROMPT = """\
以下のYouTube動画原稿が動画化された場合の視聴維持率を予測してください。

【仮タイトル】{tentative_title}
【趣旨】{purpose}

{review_concept}

━━━ 出力形式（厳守）━━━

■ 予測視聴維持率カーブ:
  導入〜1分: ○○%（理由1文）
  第1章: ○○%（理由1文）
  第2章: ○○%（理由1文）
  第3章: ○○%（理由1文）
  第4章〜ラスト: ○○%（理由1文）

■ 離脱防止の改善提案（最大4つ、効果が大きい順）:
  1. 該当箇所「引用10〜20字」→ 具体的改善内容（1-2文）→ 予測効果: +○%維持
  2. ...
  3. ...
  4. ...

※論理構造や内容の正確性には触れないでください。視聴維持率のみ。
━━━━━━━━━━━━━━━

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
        f"{CHANNEL_IDENTITY}\n"
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
OPINION_SELECTION_SYSTEM = f"""\
{CHANNEL_IDENTITY}

あなたは優秀な編集長です。
論証審査官・初見視聴者・視聴維持率コンサルタント・世論感覚チェッカー・コピーライターの5人のレビュー結果を精査し、
原稿のコンセプトに最も沿った意見だけを取捨選択してください。

重要な方針:
- すべての意見を採用してはいけません。最大6件に厳選してください。
- 複数のレビュアーが同じことを指摘している場合は1件に統合してください。
- 原稿の主張軸から逸脱する提案は必ず不採用にしてください。
- 世論感覚チェッカーが「制度側寄り」と判定した場合、その指摘は優先的に採用を検討してください。
  視聴者は国民であり、国民の実感とズレた原稿は信頼を失います。
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

━━━ 出力形式（厳守）━━━

## 採用するタイトル
コピーライターの10案から最も適切な1つを選び、その理由を1行で述べてください。

## 採用する改善意見（最大6件 — これ以上は絶対に採用しない）

| # | 出典 | 対象章 | 改善内容（1文で簡潔に） | 採用理由（1文） |
|---|------|--------|----------------------|---------------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |

※上限6件。それ以上はどんなに良い意見でも不採用。
※複数レビュアーが同じことを指摘している場合は1件に統合。
※主張軸から逸脱する提案、章の役割を変える提案は必ず不採用。

## 不採用とした主な意見（3件程度、理由付き）
- [出典] [内容要約] → 不採用理由

## リライト時の注意事項（3点以内）
※以下は必ず含めてください（固定事項）:
- 原稿の文字数（8,000文字以上）を維持すること。既存内容の削除・圧縮は禁止。
- 改善は「加筆・修正」で行い、「削って置き換える」のではない。
- 禁止ワードが含まれていないか確認すること。
━━━━━━━━━━━━━━━
"""

# --- Step 6: リライト ---
REWRITE_SYSTEM = f"""\
{CHANNEL_IDENTITY}

あなたはプロの原稿リライターです。
編集長が選定した改善意見を原稿に反映し、最終版を完成させてください。

リライトのルール:
1. 編集長が採用した改善意見のみを反映する（それ以外は変更しない）
2. 原稿の4章構成、文体、トーンは維持する
3. 読み上げ原稿としての自然さを保つ
4. 改善を加えても全体のストーリーの流れを壊さない
5. 文字数は8,000文字以上を必ず維持する。リライト前の原稿より文字数が減ることは絶対に避ける。
6. 改善意見の反映は「既存の内容を維持したまま追加・修正する」形で行う。削除は最小限に。
7. 原稿冒頭の「主張軸」を遵守する。各章は主張軸から逸脱してはならない。
8. 以下の言葉は使用禁止: 「アキレス腱」「時限爆弾」「不確実性」「構造的」「ジレンマ」「明確に」「複合的な」「単なる」「ただし」「とはいえ」「しかしながら」「もっとも」「つまり一言で言えば」
9. 逆接表現は1章につき最大1回。直後に主張を再強化する。
10. 同じ接続表現を2段落連続で使わない。
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
・原稿冒頭の「主張軸」を必ず維持してください。主張軸セクションは削除せず、そのまま残してください。
━━━━━━━━━━━━━━━━━━

採用された改善意見を反映した最終版原稿を全文出力してください。省略や要約は禁止です。
原稿の冒頭に、採用されたタイトルを「■ タイトル: 〜」の形式で記載してください。
その後に主張軸セクション、そして原稿本文を出力してください。

━━━ 反映チェックリスト（原稿の末尾に必ず出力）━━━
リライト完了後、原稿の末尾に以下を出力してください:

## 反映確認
| # | 改善意見 | 反映箇所 | 対応内容 |
|---|---------|---------|---------|
| 1 | （採用意見#1の要約） | 第○章 | （どう変更したか1文） |
| 2 | ... | ... | ... |

■ 主張軸チェック: 各章が主張軸から逸脱していないか → ✅ or ⚠️
■ 禁止ワードチェック: 使用禁止ワードが含まれていないか → ✅ or ⚠️
■ 文字数: 約○○○○文字
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# --- Step 7: ファクトチェック ---
FACT_CHECK_SYSTEM = """\
あなたはファクトチェックの専門家です。
与えられた原稿に含まれる事実・数値・主張・統計データをWeb検索で1つずつ検証してください。

重要: 原稿を全文書き直す必要はありません。
誤りが見つかった場合は「修正指示リスト」として出力してください。

以下のルールに従ってください:
1. 原稿中の具体的な数値・統計・日付・人名・組織名・法律名などを重点的に検証する
2. 検証にはWeb検索を積極的に使用し、信頼性の高い情報源を優先する
3. 検証できなかった項目はその旨を明記する
4. 原稿の文体・構成・トーンについてはコメントしない（事実の正誤のみ判断する）
"""

FACT_CHECK_USER_PROMPT = """\
以下の原稿に含まれる事実・数値・主張をWeb検索で検証してください。

━━━ 出力形式（厳守）━━━

## ファクトチェック結果
検証した項目を1つずつ以下の形式で記載:
- ✅ [検証した事実や数値] — 正確（出典: URL）
- ⚠️ [検証した事実や数値] — 要修正
  原文:「原稿中の誤りを含む該当箇所をそのまま引用」
  修正後:「正しい内容に書き換えた文」
  出典: URL
- ❓ [検証した事実や数値] — 検証不能（理由）

## 修正指示リスト
⚠️の項目のみ、以下の形式で再掲:

原文:「...」
修正後:「...」

原文:「...」
修正後:「...」

※修正不要の場合は「修正指示リスト: なし」と記載。
※原稿の全文出力は不要です。修正箇所のみ出力してください。
━━━━━━━━━━━━━━━

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


def list_history() -> list[dict]:
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


def load_history(job_id: str) -> dict | None:
    """特定の履歴データを読み込む"""
    h_path = HISTORY_DIR / f"{job_id}.json"
    if h_path.exists():
        return json.loads(h_path.read_text(encoding="utf-8"))
    return None


def format_history_as_text(data: dict) -> str:
    """履歴データをテキスト形式に整形"""
    sep = "=" * 70
    parts = [
        sep,
        " 資料→原稿化ワークフロー結果",
        f" 作成日時: {data.get('completed_at', '')}",
        f" 仮タイトル: {data.get('tentative_title', '')}",
        f" 趣旨: {data.get('purpose', '')}",
        sep,
    ]

    # 専門分野
    fields = data.get("fields", [])
    if fields:
        parts.append("\n■ 選定された専門分野:")
        for i, f in enumerate(fields, 1):
            parts.append(f"  {i}. {f.get('name', '')} - {f.get('role_description', '')}")

    # 専門家チェック
    expert_results = data.get("expert_results", [])
    for r in expert_results:
        parts.append(f"\n{sep}")
        parts.append(f" 専門家{r.get('index', 0) + 1}: {r.get('field_name', '')}の視点チェック")
        parts.append(sep)
        parts.append(r.get("review", ""))

    # 中間原稿
    draft = data.get("draft", "")
    if draft:
        parts.append(f"\n{sep}")
        parts.append(" 中間原稿 (Step 3)")
        parts.append(sep)
        parts.append(draft)

    # 批評結果
    review_results = data.get("review_results", [])
    if review_results:
        for rv in review_results:
            parts.append(f"\n{sep}")
            parts.append(f" {rv.get('name', '')}の批評")
            parts.append(sep)
            parts.append(rv.get("review", ""))

    # 意見選定結果
    selected = data.get("selected_opinions", "")
    if selected:
        parts.append(f"\n{sep}")
        parts.append(" 意見選定結果 (Step 5)")
        parts.append(sep)
        parts.append(selected)

    # リライト済み原稿
    rewritten = data.get("rewritten_draft", "")
    if rewritten:
        parts.append(f"\n{sep}")
        parts.append(" リライト済み原稿 (Step 6)")
        parts.append(sep)
        parts.append(rewritten)

    # 最終原稿
    final = data.get("final", "")
    if final:
        parts.append(f"\n{sep}")
        parts.append(" 最終原稿 - ファクトチェック済み (Step 7)")
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
# API処理
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


async def api_call_with_retry(client, event_queue: queue.Queue, label: str, **kwargs) -> object:
    """リトライ付きAPI呼び出し。429だけでなくサーバーエラー・タイムアウトにも対応。"""
    last_error = None
    for retry in range(MAX_RETRIES):
        try:
            response = await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=300,
            )
            return response
        except asyncio.TimeoutError:
            last_error = Exception("API呼び出しが5分以上応答なし")
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


async def call_with_continuation(client, model: str, system: str, messages: list[dict], max_tokens: int, event_queue: queue.Queue, label: str = "", use_search: bool = False, search_tool: dict | None = None) -> str:
    """API呼び出し＋途切れた場合に自動で続きを取得。pause_turnとmax_tokensを独立カウント。"""
    full_text = ""
    current_messages = list(messages)

    tools = [search_tool or WEB_SEARCH_TOOL] if use_search else []
    max_pause_turns = 10
    pause_turn_count = 0
    continuation_count = 0

    while True:
        current_tools = tools
        # pause_turn上限に達したらツールを無効化して文章出力を強制
        if pause_turn_count >= max_pause_turns and tools:
            current_tools = []
            event_queue.put({
                "type": "continuation",
                "message": f"{label} 検索回数上限に達したため、テキスト出力に切り替えます。",
            })

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=current_messages,
        )
        if current_tools:
            kwargs["tools"] = current_tools

        response = await api_call_with_retry(client, event_queue, label, **kwargs)

        chunk = extract_text_from_response(response)
        full_text += chunk

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "pause_turn":
            pause_turn_count += 1
            if pause_turn_count < max_pause_turns:
                event_queue.put({
                    "type": "continuation",
                    "message": f"{label} 検索結果を処理中... ({pause_turn_count}回目)",
                })
                current_messages = current_messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": "続けてください。"},
                ]
                continue
            else:
                # 上限到達 → ツール無効化してもう一回
                current_messages = current_messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": "検索は十分です。収集した情報をもとに回答を完成させてください。"},
                ]
                continue

        if response.stop_reason == "max_tokens":
            continuation_count += 1
            if continuation_count <= MAX_CONTINUATIONS:
                event_queue.put({
                    "type": "continuation",
                    "message": f"{label} 出力が長いため続きを取得中... ({continuation_count}回目)",
                })
                current_messages = current_messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": "途切れてしまいました。直前の続きから、最後まで完成させてください。重複せず、途切れた箇所のすぐ続きから書いてください。"},
                ]
                continue
            else:
                break

        # その他の stop_reason
        break

    return full_text


# --- Step 0: 自動リサーチ ---
async def auto_research_topic(client, tentative_title: str, purpose: str, model: str, event_queue: queue.Queue) -> str:
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
async def select_expert_fields(client, material: str, model: str, event_queue: queue.Queue) -> list[dict]:
    """資料を分析し、最適な3つの専門分野を自動選定"""
    event_queue.put({"type": "step", "step": 1, "message": "資料を分析し、最適な専門分野を選定中..."})

    response = await api_call_with_retry(
        client, event_queue, "分野選定",
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
async def run_expert_review(client, material: str, field: dict, index: int, model: str, event_queue: queue.Queue, use_search: bool = False) -> dict:
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
async def draft_manuscript(client, material: str, expert_results: list[dict], tentative_title: str, purpose: str, event_queue: queue.Queue) -> str:
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
        DRAFTING_MAX_TOKENS, event_queue, label="原稿化エージェント(Opus)"
    )

    event_queue.put({
        "type": "draft_done",
        "draft": draft,
    })

    return draft


# --- Step 4: 批評フェーズ ---
async def run_single_reviewer(client, draft: str, agent: dict, index: int, tentative_title: str, purpose: str, model: str, event_queue: queue.Queue) -> dict:
    """1人のレビュアーによる批評"""
    event_queue.put({"type": "reviewer_start", "index": index, "name": agent["name"]})

    start_time = time.time()

    # エージェントごとの専用プロンプトを選択
    prompt_map = {
        "論証審査官": CRITIC_USER_PROMPT,
        "初見視聴者": COMMENTATOR_USER_PROMPT,
        "視聴維持率コンサルタント": ADVISOR_USER_PROMPT,
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


async def run_review_phase(client, draft: str, tentative_title: str, purpose: str, model: str, event_queue: queue.Queue) -> list[dict]:
    """4人のレビュアーを並列実行"""
    event_queue.put({"type": "step", "step": 4, "message": "4人のレビュアーが原稿を批評中..."})

    tasks = [
        run_single_reviewer(client, draft, agent, i, tentative_title, purpose, model, event_queue)
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
async def select_opinions(client, draft: str, review_results: list[dict], tentative_title: str, purpose: str, event_queue: queue.Queue) -> str:
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
        OPINION_MAX_TOKENS, event_queue, label="意見選定(Opus)"
    )

    event_queue.put({
        "type": "opinions_selected",
        "opinions": result,
    })

    return result


# --- Step 6: リライト ---
async def rewrite_manuscript(client, draft: str, selected_opinions: str, event_queue: queue.Queue) -> str:
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
        REWRITE_MAX_TOKENS, event_queue, label="リライト(Opus)"
    )

    event_queue.put({
        "type": "rewrite_done",
        "rewritten": result,
    })

    return result


# --- Step 7: ファクトチェック ---
async def fact_check_draft(client, draft: str, event_queue: queue.Queue) -> tuple:
    """原稿をWeb検索でファクトチェックし、diff方式で修正を適用"""
    event_queue.put({"type": "step", "step": 7, "message": f"ファクトチェック中...（{DEFAULT_MODEL} + Web検索）"})

    messages = [{
        "role": "user",
        "content": FACT_CHECK_USER_PROMPT.format(draft=draft),
    }]

    result = await call_with_continuation(
        client, DEFAULT_MODEL, FACT_CHECK_SYSTEM, messages,
        FACT_CHECK_MAX_TOKENS, event_queue, label="ファクトチェック(Sonnet)",
        use_search=True, search_tool=FACT_CHECK_SEARCH_TOOL,
    )

    # ファクトチェック結果全体をレポートとして保持
    fact_report = result

    # diff方式: 「原文:」「修正後:」のペアを抽出して自動適用
    corrections = re.findall(
        r'原文[:：]\s*[「「](.+?)[」」]\s*\n\s*修正後[:：]\s*[「「](.+?)[」」]',
        result, re.DOTALL,
    )

    corrected_draft = draft
    applied_count = 0
    for original, fixed in corrections:
        original = original.strip()
        fixed = fixed.strip()
        if original and fixed and original in corrected_draft:
            corrected_draft = corrected_draft.replace(original, fixed, 1)
            applied_count += 1

    if corrections:
        event_queue.put({
            "type": "continuation",
            "message": f"ファクトチェック: {len(corrections)}件の修正指示を検出、{applied_count}件を自動適用しました。",
        })
    else:
        event_queue.put({
            "type": "continuation",
            "message": "ファクトチェック: 修正不要でした。原稿をそのまま使用します。",
        })

    event_queue.put({
        "type": "fact_check_done",
        "fact_report": fact_report,
    })

    return corrected_draft, fact_report


async def generate_reference_list(
    client, final_draft: str, fact_report: str,
    expert_results: list[dict], event_queue: queue.Queue,
) -> str:
    """最終原稿と情報源データから参考文献・出典リストを生成する"""
    event_queue.put({
        "type": "step",
        "step": "7b",
        "message": "参考文献・出典リストを生成中...",
    })

    # 専門家リサーチ結果からURL含む情報を組み立て
    expert_sources_text = ""
    for r in expert_results:
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
    )

    event_queue.put({
        "type": "reference_list_done",
        "reference_list": result,
    })

    return result


# ──────────────────────────────────────────────
# パイプライン
# ──────────────────────────────────────────────

async def run_pipeline(material: str, tentative_title: str, purpose: str, model: str, api_key: str, use_search: bool, event_queue: queue.Queue, checkpoint_id: str | None = None, auto_research: bool = False):
    """メインパイプライン（チェックポイント対応）。auto_research=True時はStep 0で資料を自動生成"""
    cp_id = checkpoint_id or str(uuid.uuid4())[:8]
    cp_data = load_checkpoint(cp_id) if checkpoint_id else None
    start_step = (cp_data.get("completed_step", 0) + 1) if cp_data else 1

    # チェックポイントから中間データを復元
    fields = cp_data.get("fields") if cp_data else None
    results = cp_data.get("expert_results") if cp_data else None
    draft = cp_data.get("draft") if cp_data else None
    review_results = cp_data.get("review_results") if cp_data else None
    selected_opinions = cp_data.get("selected_opinions") if cp_data else None
    rewritten_draft = cp_data.get("rewritten_draft") if cp_data else None

    # チェックポイント共通データ
    def cp_common():
        return {
            "material": material, "tentative_title": tentative_title,
            "purpose": purpose,
            "model": model, "use_search": use_search,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        total_start = time.time()

        if start_step > 1:
            event_queue.put({
                "type": "continuation",
                "message": f"チェックポイントからStep {start_step}より再開します（Step 1〜{start_step - 1}はスキップ）",
            })
            # 復元済みステップをUIに反映
            if fields and start_step > 1:
                event_queue.put({
                    "type": "fields_selected",
                    "fields": [{"name": f["name"], "role_description": f["role_description"]} for f in fields],
                })
                event_queue.put({"type": "restore_step", "step": 1})
            if results and start_step > 2:
                for r in results:
                    event_queue.put({"type": "expert_done", "index": r["index"], "name": r["field_name"], "review": r["review"], "elapsed": r.get("elapsed", 0)})
                event_queue.put({"type": "restore_step", "step": 2})
            if draft and start_step > 3:
                event_queue.put({"type": "draft_done", "draft": draft})
                event_queue.put({"type": "restore_step", "step": 3})
            if review_results and start_step > 4:
                for rv in review_results:
                    event_queue.put({"type": "reviewer_done", "index": rv["index"], "name": rv["name"], "review": rv["review"], "elapsed": rv.get("elapsed", 0)})
                event_queue.put({"type": "restore_step", "step": 4})
            if selected_opinions and start_step > 5:
                event_queue.put({"type": "opinions_selected", "opinions": selected_opinions})
                event_queue.put({"type": "restore_step", "step": 5})
            if rewritten_draft and start_step > 6:
                event_queue.put({"type": "rewrite_done", "rewritten": rewritten_draft})
                event_queue.put({"type": "restore_step", "step": 6})

        # Step 0: 自動リサーチ（auto_research=Trueかつ資料が空の場合）
        if auto_research and not material:
            material = await auto_research_topic(client, tentative_title, purpose, model, event_queue)
            await asyncio.sleep(STEP_COOLDOWN)

        # Step 1: 分野選定
        if start_step <= 1:
            fields = await select_expert_fields(client, material, model, event_queue)
            save_checkpoint(cp_id, {
                "completed_step": 1, "fields": fields, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # Step 2: 情報収集
        if start_step <= 2:
            num_total_experts = len(fields)
            search_label = "（Web検索あり）" if use_search else ""
            event_queue.put({"type": "step", "step": 2, "message": f"{num_total_experts}人の専門家が資料をリサーチ中...{search_label}"})
            results = []
            for batch_start in range(0, num_total_experts, PARALLEL_BATCH_SIZE):
                if batch_start > 0:
                    await asyncio.sleep(EXPERT_COOLDOWN)
                batch = fields[batch_start:batch_start + PARALLEL_BATCH_SIZE]
                tasks = [
                    run_expert_review(client, material, field, batch_start + j, model, event_queue, use_search=use_search)
                    for j, field in enumerate(batch)
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for br in batch_results:
                    if isinstance(br, Exception):
                        raise br
                    results.append(br)
            results = sorted(results, key=lambda r: r["index"])
            save_checkpoint(cp_id, {
                "completed_step": 2, "fields": fields, "expert_results": results,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # Step 3: 原稿化
        if start_step <= 3:
            draft = await draft_manuscript(client, material, results, tentative_title, purpose, event_queue)
            save_checkpoint(cp_id, {
                "completed_step": 3, "fields": fields, "expert_results": results,
                "draft": draft, **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # Step 4: 批評フェーズ
        if start_step <= 4:
            review_results = await run_review_phase(client, draft, tentative_title, purpose, model, event_queue)
            save_checkpoint(cp_id, {
                "completed_step": 4, "fields": fields, "expert_results": results,
                "draft": draft, "review_results": review_results,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # Step 5: 意見選定
        if start_step <= 5:
            selected_opinions = await select_opinions(client, draft, review_results, tentative_title, purpose, event_queue)
            save_checkpoint(cp_id, {
                "completed_step": 5, "fields": fields, "expert_results": results,
                "draft": draft, "review_results": review_results,
                "selected_opinions": selected_opinions,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # Step 6: リライト
        if start_step <= 6:
            rewritten_draft = await rewrite_manuscript(client, draft, selected_opinions, event_queue)
            save_checkpoint(cp_id, {
                "completed_step": 6, "fields": fields, "expert_results": results,
                "draft": draft, "review_results": review_results,
                "selected_opinions": selected_opinions,
                "rewritten_draft": rewritten_draft,
                **cp_common(),
            })
            await asyncio.sleep(STEP_COOLDOWN)

        # Step 7: ファクトチェック
        if start_step <= 7:
            final, fact_report = await fact_check_draft(client, rewritten_draft, event_queue)

        # Step 7b: 参考文献・出典リスト生成
        reference_list = ""
        try:
            reference_list = await generate_reference_list(
                client, final, fact_report, results, event_queue
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
            "material": material,
            "tentative_title": tentative_title,
            "purpose": purpose,
            "fields": fields,
            "expert_results": results,
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

        event_queue.put({
            "type": "complete",
            "final": final,
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
    loop.run_until_complete(run_pipeline(material, tentative_title, purpose, model, api_key, use_search, event_queue, checkpoint_id, auto_research=auto_research))
    loop.close()


# ──────────────────────────────────────────────
# Flaskルート
# ──────────────────────────────────────────────

@app.route("/")
def index():
    import pathlib
    html_path = pathlib.Path(__file__).parent / "templates" / "index.html"
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

    material = cp_data["material"]
    tentative_title = cp_data.get("tentative_title", "")
    purpose = cp_data.get("purpose", "")
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
    filename = f"原稿_{timestamp}_{job_id}.txt"
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
    port = int(os.environ.get("PORT", 8080))
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  資料→原稿化ワークフローシステム v2 (7ステップ)         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  ブラウザで開く: http://localhost:{port}                   ║
║                                                          ║
║  停止: Ctrl+C                                            ║
╚══════════════════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=port, debug=False)
