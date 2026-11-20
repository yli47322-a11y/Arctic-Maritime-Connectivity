"""Synchronize eight documented port labels across the active settlement tables."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import zipfile

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "人类聚居点/俄罗斯港口_官方来源与AMAP筛选"
RULES = ROOT / "人类聚居点/港口与沿海聚居点_规则结果"
FORMAL = ROOT / "正式成果/空间基准_5km_最终正式版_v3_全量最近出海_2026-11-04"
FIXED = ROOT / "预处理/PIOMAS_5km_标准流程_v1/固定点位"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path, text, encoding="utf-8"):
    temporary = path.with_name("." + path.name + ".updating")
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name("." + path.name + ".updating")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


suggestions = read(AUDIT / "港口与现有聚居地对应_建议标签.csv")
changes = {r["entity_id"]: r for r in suggestions
           if r["disposition"] == "保留" and r["previous_category"] == "沿海聚居点"}
assert len(changes) == 8
sources = [
    "https://base.garant.ru/6742107/",
    "https://base.garant.ru/70581512/",
    "https://www.rosmorport.ru/services/seaports/",
]
log = []


def update(record):
    key = record.get("entity_id")
    if key not in changes:
        return False
    assert record["category"] in ("沿海聚居点", "港口聚居点")
    record["category"] = "港口聚居点"
    evidence = "RU_official_port_register_or_location_link_2026-11-11"
    if evidence not in record.get("port_evidence", ""):
        record["port_evidence"] = ";".join(filter(None, [record.get("port_evidence", ""), evidence]))
    if "port_status" in record:
        record["port_status"] = "documented_port_or_quay"
    if "reason" in record and record.get("disposition") == "保留":
        record["reason"] = "距海岸不超过10km；俄罗斯官方港口登记或明确地点对应提供港口证据"
    record["official_port_name_ru"] = changes[key]["port_name_ru"]
    record["port_attribute_update_date"] = "2026-11-11"
    record["port_operation_status"] = "not_inferred_from_register_membership"
    record["port_evidence_urls"] = ";".join(sources)
    return True


csv_paths = [
    RULES / "全部点位_分类及去留.csv",
    RULES / "保留点位_港口与沿海.csv",
    ROOT / "人类聚居点/人口属性/保留4258点_补充人口属性.csv",
    FORMAL / "01_最终点位与对应/最终聚居点_4258条.csv",
    FORMAL / "01_最终点位与对应/出发点与聚居点对应_4258条.csv",
    FIXED / "聚居点_4258条.csv",
    FIXED / "聚居点与出发格对应.csv",
]
for path in csv_paths:
    records = read(path)
    before = [{k: v for k, v in row.items() if k in ("entity_id", "category", "port_evidence")}
              for row in records if row.get("entity_id") in changes]
    count = sum(update(record) for record in records)
    assert count == 8, (path, count)
    write_csv(path, records)
    log.append(dict(path=str(path.relative_to(ROOT)), changed_records=count, previous_attributes=before))

geojson_paths = [
    RULES / "保留点位.geojson",
    FORMAL / "01_最终点位与对应/最终聚居点_4258条.geojson",
    FIXED / "聚居点_4258条.geojson",
]
for path in geojson_paths:
    document = json.loads(path.read_text(encoding="utf-8"))
    count = sum(update(feature["properties"]) for feature in document["features"])
    assert count == 8, (path, count)
    atomic_text(path, json.dumps(document, ensure_ascii=False))
    log.append(dict(path=str(path.relative_to(ROOT)), changed_records=count))

all_rows = read(RULES / "全部点位_分类及去留.csv")
kept = [r for r in all_rows if r["disposition"] == "保留"]
assert len(kept) == 4258
counts = Counter(r["category"] for r in kept)
assert counts == {"港口聚居点": 272, "沿海聚居点": 3986}
write_csv(RULES / "港口聚居点.csv", [r for r in kept if r["category"] == "港口聚居点"])
write_csv(RULES / "沿海聚居点.csv", [r for r in kept if r["category"] == "沿海聚居点"])
country = read(RULES / "按国家地区汇总.csv")
for item in country:
    if item["country_code"] == "RU":
        item["港口聚居点"] = 15
        item["沿海聚居点"] = 734
write_csv(RULES / "按国家地区汇总.csv", country)

stats_path = RULES / "规则分类统计.json"
stats = json.loads(stats_path.read_text(encoding="utf-8"))
stats["categories"]["港口聚居点"] = 272
stats["categories"]["沿海聚居点"] = 3986
stats["port_attribute_update_date"] = "2026-11-11"
atomic_text(stats_path, json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
config_path = ROOT / "研究配置.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["spatial_scope"]["settlement_inclusion_rule"]["rule_classified_counts"] = stats
config["spatial_scope"]["port_attribute_update"] = dict(date="2026-11-11", updated_records=8,
                                                      ports=272, other_coastal=3986,
                                                      audit_directory=str(AUDIT.relative_to(ROOT)))
atomic_text(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")

version_path = FORMAL / "版本摘要.json"
version = json.loads(version_path.read_text(encoding="utf-8"))
version["port_attribute_update"] = dict(date="2026-11-11", updated_records=8,
                                       port_records=272, other_coastal_records=3986)
atomic_text(version_path, json.dumps(version, ensure_ascii=False, indent=2) + "\n")
manifest_path = FORMAL / "文件清单与SHA256.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for entry in manifest:
    path = FORMAL / entry["path"]
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

archive_path = FORMAL.with_suffix(".zip")
if archive_path.exists():
    temporary = archive_path.with_name("." + archive_path.name + ".updating")
    with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            relative = Path(item.filename)
            if relative.parts and relative.parts[0] == FORMAL.name:
                relative = Path(*relative.parts[1:])
            current = FORMAL / relative
            target.writestr(item, current.read_bytes() if current.is_file() else source.read(item.filename))
    os.replace(temporary, archive_path)

(AUDIT / "正式属性更新记录.json").write_text(
    json.dumps(dict(date="2026-11-11", updated_ids=list(changes), counts=dict(counts), files=log),
               ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(updated=8, counts=dict(counts), files=len(log)), ensure_ascii=False))
