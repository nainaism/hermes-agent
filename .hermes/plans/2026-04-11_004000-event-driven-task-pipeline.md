# Event-Driven Task Pipeline — 時系列・情報フロー分析

## 既存設計（project-driver SKILL.mdで実装済み）

```
T=0   PD(23m) が Todoタスクを発見
T=1   PD が Assignee を自動判定（ai-factory-project基準）
T=2   PD が ncli で Status=In Progress + Assignee=ハカセ/はなび に更新
T=3   PD が Discord に @mention メッセージを送信
        ├─ プロジェクトにThread URLがあれば → スレッド内に送信
        └─ Thread URLがなければ → #team-internal に送信
T=4   ハカセ/はなびがメッセージを受信
T=5   ???（未定義）
T=6   ???（未定義）
T=7   ???（未定義）
T=8   ???（未定義）
T=9   ???（未定義）
T=10  次回PDが Pending Review を検知 → レビュー/完了処理
```

## T=0〜T=3 は動いている

✅ project_driver_query.py が todo_tasks を取得
✅ PD が Assignee 基準で判定（実装→ハカセ、ブログ→はなび）
✅ PD が ncli で Status/Assignee 更新
✅ PD が Discord API で @mention メッセージ送信
✅ メッセージフォーマット: `<@{BOT_ID}> 📋 タスク依頼\n**{タスク名}**\n{Notionリンク}`

## T=4〜T=9 が欠けている

ハカセ/はなびの personality には「タスクを受領して自律的に進める」という**意図**は書いてあるが、**具体的な手順**がない:
- Notionリンクからどうタスクをfetchするか
- ai-factory-projectスキルをどう参照するか
- Progressにどう進捗を記録するか
- 完了時にどうStatusを更新し報告するか

## 提案: 受信側フローを補完する

### 情報フロー全体（完成形）

```
T=0   PD が Todoタスク発見
T=1   Assignee判定 → ハカセ or はなび
T=2   ncli page update → Status=In Progress, Assignee=ハカセ/はなび
T=3   Discord @mention 送信（スレッド or #team-internal）
      ↓
T=4   ハカセ/はなびがメッセージ受信
      personality + ai-factory-projectスキルでフロー認識
      ↓
T=5   ncli fetch でタスクbody取得 → Goal/AC/Context 確認
      ↓
T=6   Progressに着手追記: `YYYY-MM-DD HH:MM ハカセ/はなびが着手`
      ↓
T=7   自律的に作業実行（実装/コンテンツ作成）
      ↓
T=8   完了処理:
      ├─ AC達成 → ncli page update Status=Pending Review
      ├─ Progress追記: `YYYY-MM-DD HH:MM Pending Review`
      └─ Discord スレッドに完了報告: `✅ 完了: {タスク名}`
      ↓
T=9   （ブロック時）
      ├─ ncli page update Status=Blocked
      ├─ Progress追記: `YYYY-MM-DD HH:MM Blocked: {理由}`
      └─ Discord スレッドに報告: `🚫 ブロック: {タスク名} — {理由}`
      ↓
T=10  次回PD が Pending Review を検知 → かえでレビュー or 成田さん確認
```

### 何を変更するか

| 変更箇所 | 内容 | 理由 |
|---------|------|------|
| CTO personality | タスク受信フローを追記（5行程度） | T=4〜T=9を自律実行させる |
| CMO personality | 同上（5行程度） | 同上 |
| ai-factory-project SKILL.md | 「Discordタスク受信フロー」セクション追加 | 受信側の参照ドキュメント |

### personality追記の最小構成（CTO例）

```
## Discordタスク受信フロー

Notionリンク付きのタスク依頼メッセージを受け取った場合:
1. ai-factory-projectスキルを参照し、ncli fetch でタスクを取得
2. Goal/ACを確認し、Progressに着手を記録
3. 自律的に作業を実行
4. 完了 → Status=Pending Review に更新し、同じスレッドに完了報告
5. ブロック → Status=Blocked に更新し、同じスレッドに理由を報告
```

**詳細はai-factory-projectスキルに書くので、personalityは最小限。**

### ai-factory-project SKILL.md追加セクション

新規に「Discordタスク受信フロー」セクションを追加:
- メッセージの認識方法（Notionリンク付きタスク依頼）
- fetch → Goal確認 → 実行 → Status更新 → 報告 の具体的手順
- ncliコマンド例（既存のクイックリファレンスから参照）
- 進捗記録のフォーマット（worker-guide参照）

### 変更しないもの

- project-driver SKILL.md — 送信側の設計は完成済み
- cronジョブ — 追加なし（Discordメッセージがトリガー）
- スクリプト — 変更なし

## 懸念点

### 1. ハカセ/はなびがスキルを参照するか？
- `ai-factory-project` はグローバルスキルで全プロファイルに見えている
- personalityに「ai-factory-projectスキルを参照」と書けばLLMが skill_view するはず
- **ただし、確証がないのでテストが必要**

### 2. スレッドのメッセージ履歴が見えるか？
- HermesはDiscordのメッセージ履歴を遡れない（既知の制約）
- ただし、**自分に送られたメッセージ（@mention）は見える**
- スレッド内の@mentionも「自分宛てのメッセージ」として受信されるはず
- **ここは確認が必要**

### 3. 報告先のスレッドIDをどう知るか？
- @mentionメッセージを受信した時点で、chat_idが分かる
- そのchat_idに返信すればスレッド内報告になる
- **Hermesのsend_messageやterminal経由のDiscord APIで可能**
