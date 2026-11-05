# 第2回 基礎宿題：20エポックと10エポックの比較

状態：Colab実行待ち。学習・評価の実測値はまだない。基礎課題は未完了。

公式README、lectures/02/problem.md、answer.md、notebooks/lora_format_tuning.ipynb と、本人提供の「第二回宿題資料.pdf」（全3ページ）を確認した。PDFは設定変更後のセッション再起動、PINS維持、基礎の比較項目を指示しており、本人の依頼と一致する。締切は「原則として講義日から1週間」とあり、実際の締切日は未確認。PDF自体は再配布しない。

確認時の公式・forkコミット：`fae899a409e2c99feee9d44a3789f936f5fc64ee`。公式ノートブックと依存定義は変更していない。講師の `reference/` の値を本人の実測には使用しない。

## Colabでの操作

1. `notebooks/lec02_baseline20.ipynb` をColabにアップロードする（公開後は下のリンクも利用可能）。必要なら「ドライブにコピーを保存」。Driveが書けなければコピーせず実行して、必ずPCにダウンロードする。
2. ランタイム → ランタイムのタイプを変更 → **T4 GPU**。実行済みのセッションなら再起動する。
3. 上から最後の **7b** まで順に実行する。`USE_DRIVE=False` のまま。インストール後にColabが再起動を求めた場合は再起動して最初から実行する。PINSの版は変えない。
4. 最後に `baseline20_日時_ID.zip` がダウンロードされる。PC上のZIPに `result.json`、`eval_before.json`、`eval_after.json` があることを確認する。実行済みnotebookの自動取得に失敗した場合は、ファイル → ダウンロード → .ipynb も保存する。
5. **ZIPの保存確認後**にランタイム → セッションを再起動する。`lec02_epochs10.ipynb` を開き、上から7bまで同様に実行して `epochs10_日時_ID.zip` を保存する。20の学習済みモデルへの追加学習は禁止。GPU型もそろえる。
6. 2つのZIP（必要なら別途保存した実行済みipynbも）をこのチャットに渡す。比較・失敗原文確認・考察・GitHub更新・Q3確定を続ける。終了後はランタイムを接続解除して削除する。

- [基準20をColabで開く](https://colab.research.google.com/github/celloli/lecture-ai-engineering-2026/blob/lec02-epochs-comparison/notebooks/lec02_baseline20.ipynb)
- [変更10をColabで開く](https://colab.research.google.com/github/celloli/lecture-ai-engineering-2026/blob/lec02-epochs-comparison/notebooks/lec02_epochs10.ipynb)

切断・エラー時は実行した範囲とエラーを保存して相談する。別条件を途中checkpointから始めない。結果を退避せずセッションを削除しない。任意のセクション8は今回は実行しない。

## 比較条件と計測

| 項目 | 基準 | 変更 |
|---|---|---|
| num_train_epochs | 20（公式現行値） | 10 |
| ステップ数 | 40（予定） | 20（予定） |
| モデル | Qwen/Qwen3.5-0.8B | 同じ |
| revision | 2fc06364715b967f1860aea9cf38778875588b17 | 同じ |
| seed / lora_r / lora_alpha / lr | 42 / 16 / 32 / 0.0002 | 同じ |
| warmup_steps / scheduler | 6 / cosine | 同じ指定 |
| train / valid / test | 14 / 7 / 21件 | 同じ |
| 生成 | do_sample=False、max_new_tokens=96 | 同じ |
| 重み / 混合精度 | float32 / USE_AMP=False | 同じ |

PINS：transformers=5.18.0、peft=0.21.1、trl=1.14.2、accelerate=1.15.0、datasets=5.1.0。torchは公式同様Colab同梱版を使用し、両条件の実際の版も比較する。任意のbitsandbytes導入セルは含めない。バッチサイズ、勾配累積、dropout、対象層、プロンプト、採点関数も公式と同じ。

セクション6の表示メタデータだけを1か所修正した。公式は `q in gold` でFAQ内か判定するが、最後の対照質問はtrainに含まれtest用goldにはないため誤ってFalseになる。派生版では最後の質問をFAQ内と記録する。モデルへの入力・生成・分類・testの採点は変更しておらず、両条件に同じ修正を適用する。

エポック変更で総ステップ数が変わるため、cosineの学習率軌跡やwarmupの割合も変わる。同じ軌跡で単純に途中打ち切りする比較ではない。各条件1回・同じseedのみで一般的な優劣は断定しない。

時間は `trainer.train()` の前後でCUDA完了を待ち `time.perf_counter()` で計測。train内のcheckpoint保存は含み、モデル読込・Trainer構築・終了後のadapter保存・評価は除く。最終lossは最後のTrainerログ `loss`（最終10ステップ区間の平均）とstepを記録する。全学習平均 `train_loss_all_steps` と区別する。計測値が不足した場合は推測せず比較処理を停止する。

新規セッション確認・学習セルの二重実行防止・一意の実行IDを追加。保存先は `/content/aie02/条件名_実行ID/` でDrive不要。ZIPには許可したJSON・データ・実行済みipynbだけを梱包し、adapter・モデル・checkpointは含めない。

## 回収後の処理

Python標準ライブラリのみ。講師参考結果は自動取り込みしない。ZIPのパス、設定差、PINS、実際の版、データ指紋、評価原文と採点、lossログと完了stepを検査する。

```console
python scripts/compare_lec02.py --baseline /path/to/baseline20_ID.zip --changed /path/to/epochs10_ID.zip --output lectures/02/results/measured
```

出力先は新規フォルダを指定する（既存結果は上書きしない）。生成するもの：

- `baseline20/` と `epochs10/`：各条件のJSON・評価原文・設定・学習ログ・版・データ・実行済みipynb
- `comparison.md` / `.csv` / `.json`：実測比較表、退行と失敗原文、知識質問の前後分類、留意点
- `answer_draft.md`：実測解答案。原文を読んで本人の考察を追記し、`lectures/02/answer.md` に反映
- `omnicampus_draft.md`：実測のQ1〜Q3案。Q3はPythonのlenで400文字以内を検査。本人確認後 `OMNICAMPUS.md` に反映

実行済みipynbがZIPにない場合は別途保存したものを各条件フォルダへ追加する。公開前にキー・認証情報・個人情報の混入を確認する。GPU型・Python/CUDA・学習前出力の差は警告として表示する。差がある場合は制約を明記するか条件をそろえて再実行する。

## GitHub保存と理解

保存先は本人確認済みの公開fork [celloli/lecture-ai-engineering-2026](https://github.com/celloli/lecture-ai-engineering-2026)。作業ブランチ `lec02-epochs-comparison` → **同じforkのmain** にPRを作る。公式matsuolabにはPRを作らない。未実行段階はDraft PRとし、実測・考察・Q3確認後にレビュー可能にする。

再生成：`python scripts/prepare_lec02_comparison.py`。派生ノートブックが既存の場合は停止するので、元の版を保管した新規作業コピーで実施する。

- LoRA：基盤モデルを固定し、追加した小さな行列だけを学習する。今回は量子化なしでQLoRAではない。
- エポック：学習データ全体を何周するか。20→10で学習量は減るが時間が必ず半分になるとは限らない。
- 形式遵守率：JSONのキー・型が正しい割合。
- 内容正解率：needs_reviewと必須語句での自動判定。形式違反でもJSON部分が読めれば別に内容を採点する。完全な意味評価ではない。
- 退行：学習前は形式と内容の両方で正解だった質問が、学習後に不合格になること。
- 比較の読み方：前後の変化、20と10の差、具体的失敗を順に見る。形式が整っても根拠判断まで保証されない。セクション6のFAQ内の対照質問では、正しい根拠による断定は失敗ではない。

コードの実行と本人が仕組みを説明できることは別。現時点は実行準備済み、本人の理解は未確認。
