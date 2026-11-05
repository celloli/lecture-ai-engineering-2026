"""Create two minimally instrumented notebooks from the unmodified official notebook."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "fae899a409e2c99feee9d44a3789f936f5fc64ee"
SOURCE = ROOT / "notebooks/lora_format_tuning.ipynb"
official = json.loads(SOURCE.read_text(encoding="utf-8"))
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def cell(kind, text, ident):
    c = {"cell_type": kind, "id": ident, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


MEASURE = '''# 計測範囲: trainer.train() のみ。モデル読込・Trainer構築・adapter保存・評価を除く。
# train() 内のチェックポイント保存は含む。CUDA完了を前後で待つ。
if globals().get("_TRAINING_STARTED", False):
    raise RuntimeError("学習セルの再実行は禁止。結果を退避してセッションを再起動してください。")
_TRAINING_STARTED = True
if torch.cuda.is_available(): torch.cuda.synchronize()
train_start = time.perf_counter()
train_output = trainer.train()
if torch.cuda.is_available(): torch.cuda.synchronize()
train_seconds = time.perf_counter() - train_start
loss_logs = [r for r in trainer.state.log_history if "loss" in r]
training_metrics = {
    "train_seconds": train_seconds,
    "timing_scope": "trainer.train() including checkpoint saving; CUDA synchronized; excludes model loading, Trainer construction, final adapter saving and evaluation",
    "final_logged_loss": loss_logs[-1]["loss"] if loss_logs else None,
    "final_loss_step": loss_logs[-1]["step"] if loss_logs else None,
    "loss_definition": "last Trainer logged loss (mean over the final logging interval), not a single minibatch loss",
    "train_loss_all_steps": train_output.metrics.get("train_loss"),
    "global_step": trainer.state.global_step,
    "epoch": trainer.state.epoch,
    "trainer_metrics": train_output.metrics,
}
(WORK / "training_metrics.json").write_text(json.dumps(training_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
(WORK / "trainer_log_history.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
print(training_metrics)
'''

SAVE_EXTRA = '''
# 生出力を残して複数の失敗型を記録する（同じ回答に複数の問題があり得る）。
failures = []
for r in after_rows:
    if r["both_ok"]: continue
    types = []
    if not r["format_ok"]: types.append("形式の崩れ")
    ok, _, obj = check_format(r["raw"])
    if ok:
        if r["gold_needs_review"] and not obj["needs_review"]:
            types.append("根拠なしへの断定")
        elif not r["gold_needs_review"] and obj["needs_review"]:
            types.append("根拠ありへの不要な確認要求")
        elif not r["content_ok"]:
            types.append("必須語句の不一致（原文を要確認）")
    failures.append({**r, "failure_types": types})
import hashlib, platform, importlib.metadata
data_hashes = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest()
               for name in ["train.jsonl", "valid.jsonl", "test.jsonl", "gold.json"]}
result.update({
    "condition": CONDITION, "run_id": RUN_ID,
    "upstream_commit": UPSTREAM_COMMIT, "official_notebook_sha256": OFFICIAL_NOTEBOOK_SHA256,
    "experiment_protocol_sha256": EXPERIMENT_PROTOCOL_SHA256,
    "training": training_metrics, "failures": failures, "data_sha256": data_hashes,
    "pins": PINS,
    "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, accelerate, datasets, torch, pd]},
    "runtime": {"python": platform.python_version(), "gpu": gpu_info,
                "cuda": torch.version.cuda, "platform": platform.platform()},
    "training_completed": True,
    "fresh_base_session": True,
})
assert all(result["versions"][k] == v for k, v in PINS.items()), "PINSと実際の版が異なります"
assert trainer.state.epoch == CONFIG["num_train_epochs"], "学習が完了していません"
(WORK / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
(WORK / "config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
(WORK / "knowledge.json").write_text(json.dumps(knowledge_rows, ensure_ascii=False, indent=2), encoding="utf-8")
(WORK / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
# URL・トークン等を含むpip freezeではなく配布名と版のみ記録。
versions = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]})
(WORK / "library_versions.json").write_text(json.dumps(dict(versions), indent=2), encoding="utf-8")
'''

EXPORT = '''# 再起動前にこのセルを実行してZIPのダウンロード完了を確認する。
from google.colab import files, _message
import zipfile
try:
    snap = _message.blocking_request("get_ipynb", timeout_sec=30)["ipynb"]
    (WORK / "executed_notebook.ipynb").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
except Exception as exc:
    print("実行済みnotebookの自動取得に失敗:", type(exc).__name__)
    print("ファイル → ダウンロード → .ipynb を別途保存し、ZIPと一緒に渡してください。")
archive = WORK.parent / f"{CONDITION}_{RUN_ID}.zip"
# モデル重み・adapter・ckptは含めず、明示した実験記録だけを梱包。
names = ["result.json", "eval_before.json", "eval_after.json", "training_metrics.json",
         "trainer_log_history.json", "config.json", "knowledge.json", "failures.json",
         "library_versions.json", "train.jsonl", "valid.jsonl", "test.jsonl", "gold.json"]
if (WORK / "executed_notebook.ipynb").exists(): names.append("executed_notebook.ipynb")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for name in names: z.write(WORK / name, f"{CONDITION}/{name}")
print("保存先:", archive, "bytes:", archive.stat().st_size)
files.download(str(archive))
print("PCへの保存完了を確認してからセッションを再起動してください。")
'''

# 共通実験コード（設定セルを除く）の指紋。両条件の採点・学習・生成が同一か確認する。
protocol_hash = hashlib.sha256(("".join("".join(c["source"]) for i, c in enumerate(official["cells"][:20])
    if c["cell_type"] == "code" and i != 5) + MEASURE + SAVE_EXTRA).encode()).hexdigest()

for epochs, condition in [(20, "baseline20"), (10, "epochs10")]:
    nb = copy.deepcopy(official)
    nb["cells"] = nb["cells"][:20]  # 任意の4bit実験は今回の基礎比較に含めない。
    intro = f'''# 基礎宿題: {condition}（{epochs} epochs）

公式コミット `{UPSTREAM_COMMIT}` から派生。PINS・データ・モデル・学習/生成コードを維持。
2条件の学習設定の差は num_train_epochs のみです。保存先は条件別・実行別に分けます。
新しいT4セッションで上から最後まで実行してください。既存モデルからの追加学習は禁止。
最後にZIPをPCへダウンロードし、保存を確認してから再起動してください。
自動で実行済みノートブックを取得できないときは「ファイル → ダウンロード → .ipynb」も保存します。
10エポックでは20ステップ。warmup_steps=6とcosine指定は共通ですが、総ステップ数に応じて実際の学習率軌跡も変わります。
元の説明中の「20エポック・40ステップ・所要時間」は公式基準の説明で、実測ではありません。
'''
    guard = '''if any(k in globals() for k in ("model", "tuned", "trainer", "_TRAINING_STARTED")):
    raise RuntimeError("結果をダウンロードしてセッションを再起動し、上から実行してください。")
'''
    s = "".join(nb["cells"][5]["source"])
    assert '"num_train_epochs": 20' in s
    s = s.replace('"num_train_epochs": 20', f'"num_train_epochs": {epochs}')
    s = s.replace('"/content/qlora_lab"', 'f"/content/aie02/{CONDITION}_{RUN_ID}"')
    prefix = f'''from datetime import datetime, timezone
from uuid import uuid4
CONDITION = {condition!r}
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
UPSTREAM_COMMIT = {UPSTREAM_COMMIT!r}
OFFICIAL_NOTEBOOK_SHA256 = {source_hash!r}
EXPERIMENT_PROTOCOL_SHA256 = {protocol_hash!r}
'''
    nb["cells"][5]["source"] = (prefix + s).splitlines(keepends=True)
    # 同じ評価データを再計算できるようvalidも保存。
    nb["cells"][7]["source"] += ['\n_ = (WORK / "valid.jsonl").write_text("\\n".join(json.dumps(r, ensure_ascii=False) for r in valid), encoding="utf-8")\n']
    s = "".join(nb["cells"][12]["source"])
    start = s.index("t0 = time.time()")
    end = s.index('ADAPTER_DIR =')
    nb["cells"][12]["source"] = (s[:start] + MEASURE + s[end:]).splitlines(keepends=True)
    # 公式はq in goldでFAQ内を判定するが、対照質問はtrainにありtestのgoldには無い。
    # 生成・採点には触れず、この表示メタデータだけを修正する。
    s = "".join(nb["cells"][17]["source"])
    nb["cells"][17]["source"] = s.replace('"FAQ にある": q in gold',
        '"FAQ にある": q == KNOWLEDGE_QUESTIONS[-1]').splitlines(keepends=True)
    nb["cells"][19]["source"] += SAVE_EXTRA.splitlines(keepends=True)
    nb["cells"] = [cell("markdown", intro, "comparison-guide"), cell("code", guard, "fresh-session-guard")] + nb["cells"]
    nb["cells"].extend([cell("markdown", "## 7b. 結果をZIPでダウンロード（再起動する前に必須）", "export-guide"), cell("code", EXPORT, "export-results")])
    dest = ROOT / f"notebooks/lec02_{condition}.ipynb"
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(dest)
