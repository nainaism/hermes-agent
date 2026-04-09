# [SILENT] 表記揺れ対応プラン

## Goal
`[SILENT]` マーカーに加えて、日本語表記揺れ（`[サイレント]` `[さいれんと]` `[静か]` `[Silent]` など）でも配送抑制が機能するようにする。

## Current Context
`[SILENT]` 検出は3箇所で個別に実装されており、それぞれ検出ロジックが異なる：

| ファイル | 行 | 現在の検出方法 | 大小判定 |
|----------|-----|----------------|----------|
| `cron/scheduler.py` | L745 | `SILENT_MARKER in deliver_content.strip().upper()` | ✅ 大小 |
| `gateway/platforms/base.py` | L1192 | `"[SILENT]" in response` | ❌ 大小 |
| `gateway/run.py` | L3046 | `"[SILENT]" in response` | ❌ 大小 |
| `gateway/builtin_hooks/boot_md.py` | L60 | `"[SILENT]" not in response` | ❌ 大小 |

`scheduler.py` だけ `.upper()` で大小を吸収しているが、他はリテラル一致。またいずれも日本語表記揺れには対応していない。

## Approach
共通ユーティリティ関数 `_is_silent(response)` を `scheduler.py` に定義し、4箇所すべてでそれをimportして使う。新しい表記パターンは正規表現で1箇所に集約する。

### 対応パターン
```python
_SILENT_RE = re.compile(
    r"\[\s*"                    # [
    r"(?:silent|サイレント|さいれんと|静か|黙れ|shizuka)"  # aliases
    r"\s*\]",                   # ]
    re.IGNORECASE
)
```

括弧内の空白は許容（`[ SILENT ]` 等）。

## Step-by-Step Plan

### Step 1: `cron/scheduler.py` — 共通関数の定義
- `SILENT_MARKER` 定数を残す（後方互換・テスト用）
- 新たに `is_silent_response(text: str) -> bool` 関数を定義
- L745 の検出ロジックを `is_silent_response()` に差し替え
- テスト内の `SILENT_MARKER` 参照は維持（log message用なので問題なし）

### Step 2: `gateway/platforms/base.py` — import & 差し替え
- `from cron.scheduler import is_silent_response` を追加
- L1192 の `"[SILENT]" in response` を `is_silent_response(response)` に差し替え
- log message内の `[SILENT]` 文字列表記はそのまま（人間向けログなのでOK）

### Step 3: `gateway/run.py` — import & 差し替え
- `from cron.scheduler import is_silent_response` を追加
- L3046 の `"[SILENT]" in response` を `is_silent_response(response)` に差し替え

### Step 4: `gateway/builtin_hooks/boot_md.py` — import & 差し替え
- `from cron.scheduler import is_silent_response` を追加
- L60 の `"[SILENT]" not in response` を `not is_silent_response(response)` に差し替え

### Step 5: テストの更新
- `tests/cron/test_scheduler.py` の `TestSilentDelivery` クラスに日本語表記のテストケースを追加：
  - `test_silent_japanese_katakana`: `[サイレント]`
  - `test_silent_japanese_hiragana`: `[さいれんと]`
  - `test_silent_japanese_kanji`: `[静か]`
  - `test_silent_with_spaces`: `[ SILENT ]`
  - `test_not_silent_similar`: `[サイレントモード]` は**誤検知しない**こと（単語境界確認）

### Step 6: 既存テストの回帰確認
```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate
python -m pytest tests/cron/test_scheduler.py::TestSilentDelivery -v
python -m pytest tests/cron/test_scheduler.py::TestBuildJobPromptSilentHint -v
```

## Files to Change
| ファイル | 変更内容 |
|----------|----------|
| `cron/scheduler.py` | `is_silent_response()` 追加、L745 差し替え |
| `gateway/platforms/base.py` | import追加、L1192 差し替え |
| `gateway/run.py` | import追加、L3046 差し替え |
| `gateway/builtin_hooks/boot_md.py` | import追加、L60 差し替え |
| `tests/cron/test_scheduler.py` | 日本語表記テストケース追加 |

## Risks & Tradeoffs
- **誤検知リスク**: `[静か]` は一般的な言葉なので、LLMの通常レスポンスに偶然含まれる可能性が低いがゼロではない → 正規表現で `[]` で囲まれた形式に限定することで緩和
- **`[サイレントモード]` のような派生形**: パターンは完全一致（`silent` 単語のみ）なので誤検知しない
- **後方互換**: `[SILENT]` 従来動作はすべて維持。破壊的変更なし
- **`cron.scheduler` → `gateway` のimport方向**: `cron` が `gateway` に依存していないので逆方向importは問題なし（`gateway` が `cron` をimportする）

## Open Questions
- 追加する表記エイリアスはこれで十分か？追加したいものがあれば Step 1 の正規表現に追記するだけで対応可能
