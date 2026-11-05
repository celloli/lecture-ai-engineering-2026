"""Validate and compare actual Colab result bundles; Python standard library only.

Never reads the instructor reference directory implicitly. Never invents missing metrics.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

FILES = {"result.json", "eval_before.json", "eval_after.json", "training_metrics.json",
         "trainer_log_history.json", "config.json", "knowledge.json", "failures.json",
         "library_versions.json", "train.jsonl", "valid.jsonl", "test.jsonl", "gold.json",
         "executed_notebook.ipynb"}


def require(ok, message):
    if not ok: raise ValueError(message)


def read_bundle(path, condition):
    if path.is_dir():
        blobs = {n: (path / n).read_bytes() for n in FILES if (path / n).is_file()}
    else:
        with zipfile.ZipFile(path) as z:
            require(sum(i.file_size for i in z.infolist()) < 50_000_000, "ZIPが大きすぎます")
            names = [f"{condition}/{n}" for n in FILES]
            require(len(z.namelist()) == len(set(z.namelist())), "ZIPに重複名があります")
            require(all(n in names for n in z.namelist()), "ZIPに想定外のファイルがあります")
            blobs = {n: z.read(f"{condition}/{n}") for n in FILES if f"{condition}/{n}" in z.namelist()}
    require(FILES - {"executed_notebook.ipynb"} <= blobs.keys(), f"{condition}: 必須記録が不足")
    r = json.loads(blobs["result.json"])
    require(r.get("condition") == condition, "条件名が一致しません")
    require(r.get("executed_training_myself") is True and r.get("training_completed") is True,
            "本人による学習完了記録がありません。参考結果はこの比較に流用できません")
    require(r.get("fresh_base_session") is True, "新規セッションの記録がありません")
    require(r["training"] == json.loads(blobs["training_metrics.json"]), "学習計測記録が不一致")
    require(r["knowledge"] == json.loads(blobs["knowledge.json"]), "知識質問記録が不一致")
    require(len(r["knowledge"]) == 4, "知識質問は4問必要です")
    require({k:v for k,v in json.loads(blobs["config.json"]).items() if k not in ("work_dir", "drive_dir")} == r["config"], "設定記録が不一致")
    require(all(r["versions"][k] == v for k,v in r["pins"].items()), "PINSに不一致")
    for name, digest in r["data_sha256"].items():
        require(name in {"train.jsonl", "valid.jsonl", "test.jsonl", "gold.json"}, "未知のデータ名")
        require(hashlib.sha256(blobs[name]).hexdigest() == digest, f"{name}: データ指紋不一致")
    require(set(r["data_sha256"]) == {"train.jsonl", "valid.jsonl", "test.jsonl", "gold.json"}, "データ指紋不足")
    gold = json.loads(blobs["gold.json"])
    evaluations = {}
    for phase in ("before", "after"):
        ev = json.loads(blobs[f"eval_{phase}.json"])
        require(ev["summary"] == r[phase], "評価要約が不一致")
        rows = ev["rows"]
        require(len(rows) == 21 and len({x["question"] for x in rows}) == 21, "評価は固有の21問必要です")
        require(set(gold) == {x["question"] for x in rows}, "評価質問が正解データと不一致")
        for x in rows:
            try: obj = json.loads(x["raw"])
            except (ValueError, TypeError): obj = None
            fmt = (isinstance(obj, dict) and set(obj) == {"answer", "needs_review"}
                   and isinstance(obj["answer"], str) and isinstance(obj["needs_review"], bool))
            loose = obj
            if not fmt:
                m = re.search(r"\{.*\}", x["raw"], re.S)
                try: loose = json.loads(m.group(0)) if m else None
                except ValueError: loose = None
            g = gold[x["question"]]
            content = (isinstance(loose, dict) and loose.get("needs_review") == g["needs_review"]
                       and all(w in loose.get("answer", "") for w in g["must_include"]))
            require(x["format_ok"] == fmt and x["content_ok"] == content and x["both_ok"] == (fmt and content), "生出力と採点結果が不一致")
            require(x["gold_needs_review"] == g["needs_review"], "正解フラグ不一致")
        for metric, field in [("format_rate", "format_ok"), ("content_rate", "content_ok"), ("both_rate", "both_ok")]:
            require(math.isclose(sum(x[field] for x in rows)/21, r[phase][metric]), "評価率が生データと不一致")
        evaluations[phase] = rows
    before = {x["question"]: x for x in evaluations["before"]}
    regressed = [x["question"] for x in evaluations["after"] if before[x["question"]]["both_ok"] and not x["both_ok"]]
    require(regressed == r["regressed"], "退行の記録が不一致")
    t = r["training"]
    epochs = 20 if condition == "baseline20" else 10
    require(r["config"]["num_train_epochs"] == epochs and t["epoch"] == epochs and t["global_step"] == epochs*2, "学習量が条件と不一致")
    for k in ("final_logged_loss", "train_seconds"):
        require(isinstance(t.get(k), (float, int)) and math.isfinite(t[k]) and t[k] >= 0, f"{k}: 実測値が不足")
    logs = [x for x in json.loads(blobs["trainer_log_history.json"]) if "loss" in x]
    require(logs and logs[-1]["loss"] == t["final_logged_loss"] and logs[-1]["step"] == t["final_loss_step"] == t["global_step"], "最終lossとログが不一致")
    return r, blobs, evaluations


def pct(x): return f"{x:.2%}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--changed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="新規フォルダのみ。既存の結果を上書きしない")
    p.add_argument("--repository-url", default="https://github.com/celloli/lecture-ai-engineering-2026")
    a = p.parse_args()
    require(not a.output.exists(), "出力先が存在します。新しいフォルダ名を指定してください")
    b, bb, be = read_bundle(a.baseline, "baseline20")
    c, cb, ce = read_bundle(a.changed, "epochs10")
    config_diff = {k: [b["config"].get(k), c["config"].get(k)] for k in b["config"].keys() | c["config"].keys() if b["config"].get(k) != c["config"].get(k)}
    require(config_diff == {"num_train_epochs": [20, 10]}, f"エポック数以外に設定差: {config_diff}")
    for k in ("data_sha256", "pins", "versions", "upstream_commit", "official_notebook_sha256", "experiment_protocol_sha256"):
        require(b[k] == c[k], f"{k} が異なります。条件をそろえて再実行してください")
    require(b["run_id"] != c["run_id"], "別実行の記録が必要です")
    warnings = []
    if b["runtime"] != c["runtime"]: warnings.append("実行環境情報に差があります。GPU型・Python・CUDAの差を確認してから時間を比較してください。")
    if [x["raw"] for x in be["before"]] != [x["raw"] for x in ce["before"]]:
        warnings.append("学習前の出力に差があります。設定・環境を再確認し、この差も考察に記載してください。")
    for label, blobs in [("baseline20", bb), ("epochs10", cb)]:
        if "executed_notebook.ipynb" not in blobs: warnings.append(f"{label}: 実行済みipynbは別途回収してください。")
    rows = []
    for r in (b,c):
        rows.append({"条件": r["condition"], "num_train_epochs": r["config"]["num_train_epochs"],
            "形式_前": pct(r["before"]["format_rate"]), "形式_後": pct(r["after"]["format_rate"]),
            "内容_前": pct(r["before"]["content_rate"]), "内容_後": pct(r["after"]["content_rate"]),
            "最終ログloss": r["training"]["final_logged_loss"], "学習秒": round(r["training"]["train_seconds"],3),
            "退行数": len(r["regressed"]), "残った失敗数": len(r["failures"])})
    headers = list(rows[0])
    table = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n"
    table += "\n".join("| " + " | ".join(str(r[k]) for k in headers) + " |" for r in rows)
    narrative = ("各条件1回、同じseedの結果であり、一般的な優劣は断定しない。"
        "エポック数の変更に伴って総ステップ数とcosine学習率の推移も変わる。"
        "最終lossは最終10ステップ区間のログ平均で、学習全体平均lossとは区別する。"
        "形式遵守率はJSONのキーと型、内容正解率はneeds_reviewと必須語句による自動判定であり、文面の完全な意味的正しさを保証しない。")
    detail = ""
    for r in (b,c):
        detail += f"\n\n## {r['condition']}\n\n退行: {json.dumps(r['regressed'], ensure_ascii=False)}\n\n"
        detail += "失敗例（原文と自動分類。文面は目視で確認）:\n\n"
        for f in r["failures"]:
            detail += f"- 質問: {f['question']} / 分類: {', '.join(f['failure_types'])} / 出力: {json.dumps(f['raw'],ensure_ascii=False)}\n"
        if not r["failures"]: detail += "- 評価21問で残った失敗なし。この評価範囲に限る。\n"
        detail += "\n知識質問（FAQにある対照質問は断定が直ちに失敗を意味しない）:\n\n"
        for k in r["knowledge"]:
            detail += f"- {k['質問']} / FAQ内={k['FAQ にある']} / {k['学習前の分類']} → {k['学習後の分類']} / 前={json.dumps(k['学習前の出力'],ensure_ascii=False)} / 後={json.dumps(k['学習後の出力'],ensure_ascii=False)}\n"
    ft = ("今回測定した形式と内容の率、および上記の失敗原文を分けて読む。"
          "定型の出力形式や回答スタイルを多数の例でそろえたい用途はファインチューニングの候補になる。"
          "頻繁に更新される締切・人名などの事実や、根拠のない質問への確実な判断を今回の小規模学習だけに任せるのは適さない。"
          "FAQ検索や最新資料の参照、確認要求の仕組みを併用する。"
          "実測で形式だけが改善したか、内容も改善したか、断定・不要な確認要求が残ったかを上記の数字と原文から確認し、本人の言葉で追記する。")
    q3 = (f"公式LoRAノートブックのエポック数だけを20から10に変更し、別セッションで同じ基盤モデルから学習した。"
          f"学習後の形式遵守率は20回条件{pct(b['after']['format_rate'])}、10回条件{pct(c['after']['format_rate'])}、"
          f"内容正解率は各{pct(b['after']['content_rate'])}、{pct(c['after']['content_rate'])}だった。"
          "学習時間と最終ログlossも記録し、退行・失敗例・知識質問の生出力を保存した。形式と内容を分けて比較し、各1回の結果から一般的な優劣は断定しない。")
    require(len(q3) <= 400, "Q3が400文字を超えています")
    a.output.mkdir(parents=True)
    for label, blobs in [("baseline20",bb),("epochs10",cb)]:
        dest = a.output / label
        dest.mkdir()
        for n,data in blobs.items(): (dest/n).write_bytes(data)
    (a.output/"comparison.json").write_text(json.dumps({"rows":rows,"config_diff":config_diff,"warnings":warnings},ensure_ascii=False,indent=2),encoding="utf-8")
    with (a.output/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=headers); w.writeheader(); w.writerows(rows)
    report = "# 実測比較\n\n"+table+"\n\n"+narrative+"\n\n"+"\n".join(warnings)+detail
    (a.output/"comparison.md").write_text(report,encoding="utf-8")
    (a.output/"answer_draft.md").write_text("# 第2回 解答（実測から生成・本人確認前）\n\n## 演習\n\n両条件で本人が学習を実行（executed_training_myself=true）。設定・版・計測範囲は各result.json参照。\n\n"+report+"\n\n## 宿題\n\n取り組んだレベル：基礎。変更した学習設定：num_train_epochs=20→10。\n\n"+ft+"\n\n## 理解の確認\n\nコードの実行と仕組みの理解を分けて記録する。本人が説明できた内容：未確認。\n",encoding="utf-8")
    (a.output/"omnicampus_draft.md").write_text(f"# OmniCampus回答案（目視確認後に確定）\n\nQ1: 基礎（両条件の実測と考察の確認後）\n\nQ2: {a.repository_url}\n\nQ3（{len(q3)}文字、本文のみ。Python lenで計測）:\n\n{q3}\n\n生出力・理解・PRの内容を確認してから提出。OmniCampusへの入力・送信は本人が行う。\n",encoding="utf-8")
    print(table)
    print("Q3文字数:",len(q3),"; 保存先:",a.output)


if __name__ == "__main__": main()
