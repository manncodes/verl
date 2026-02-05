"""Scan the structured output dataset for schema edge cases that could trip up the reward function."""

import json
import sys
from pathlib import Path

import pandas as pd


def check_field_schema(schema, path="", issues=None):
    """Recursively check a JSON schema for edge cases."""
    if issues is None:
        issues = []

    if not isinstance(schema, dict):
        issues.append(f"  {path or '<root>'}: schema is {type(schema).__name__}, not dict")
        return issues

    properties = schema.get("properties", {})
    for field_name, field_schema in properties.items():
        field_path = f"{path}.{field_name}" if path else field_name

        # Check 1: field_schema is not a dict (list, string, etc.)
        if not isinstance(field_schema, dict):
            issues.append(f"  {field_path}: field_schema is {type(field_schema).__name__}: {repr(field_schema)[:100]}")
            continue

        # Check 2: type is a list (union type like ["string", "null"])
        field_type = field_schema.get("type")
        if isinstance(field_type, list):
            issues.append(f"  {field_path}: union type {field_type}")

        # Check 3: anyOf / oneOf / allOf (may produce unexpected structures)
        for combo_key in ("anyOf", "oneOf", "allOf"):
            if combo_key in field_schema:
                issues.append(f"  {field_path}: has {combo_key} with {len(field_schema[combo_key])} variants")

        # Check 4: $ref (schema references)
        if "$ref" in field_schema:
            issues.append(f"  {field_path}: has $ref = {field_schema['$ref']}")

        # Recurse into nested objects
        if field_type == "object" or (isinstance(field_type, list) and "object" in field_type):
            check_field_schema(field_schema, path=field_path, issues=issues)

        # Recurse into array items
        if field_type == "array" or (isinstance(field_type, list) and "array" in field_type):
            items_schema = field_schema.get("items", {})
            if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                check_field_schema(items_schema, path=f"{field_path}[]", issues=issues)
            elif isinstance(items_schema, list):
                issues.append(f"  {field_path}.items: items is a list (tuple validation): {repr(items_schema)[:100]}")

    return issues


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "/fsxp2/qpn744/data/structured_outputs/structured_output_train.parquet",
        "/fsxp2/qpn744/data/structured_outputs/structured_output_val.parquet",
    ]

    total_rows = 0
    rows_with_issues = 0
    all_issue_types = {}

    for fpath in files:
        if not Path(fpath).exists():
            print(f"SKIP: {fpath} not found")
            continue

        print(f"\n{'='*80}")
        print(f"Scanning: {fpath}")
        print(f"{'='*80}")

        df = pd.read_parquet(fpath)
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Try to find the schema column
        schema_col = None
        for col in ["schema_str", "ground_truth", "reward_model", "extra_info"]:
            if col in df.columns:
                schema_col = col
                break

        if schema_col is None:
            print(f"No obvious schema column found. Checking all string columns...")
            # Try each column to see if it contains JSON schemas
            for col in df.columns:
                sample = df[col].iloc[0] if len(df) > 0 else None
                if isinstance(sample, str) and ('"type"' in sample or '"properties"' in sample):
                    schema_col = col
                    print(f"  Found candidate schema column: {col}")
                    break
                elif isinstance(sample, dict):
                    if "schema" in sample or "schema_str" in sample or "properties" in sample:
                        schema_col = col
                        print(f"  Found candidate schema column (dict): {col}")
                        break

        if schema_col is None:
            print("Could not identify schema column. Dumping first row for inspection:")
            if len(df) > 0:
                for col in df.columns:
                    val = df[col].iloc[0]
                    print(f"  {col} ({type(val).__name__}): {repr(val)[:200]}")
            continue

        print(f"Using schema column: {schema_col}")
        print()

        for idx, row in df.iterrows():
            total_rows += 1
            raw = row[schema_col]

            # Extract schema from various formats
            schema = None
            try:
                if isinstance(raw, str):
                    parsed = json.loads(raw)
                elif isinstance(raw, dict):
                    parsed = raw
                else:
                    print(f"Row {idx}: unexpected type {type(raw).__name__}")
                    continue

                if isinstance(parsed, dict):
                    if "schema_str" in parsed:
                        schema = json.loads(parsed["schema_str"])
                    elif "schema" in parsed:
                        s = parsed["schema"]
                        schema = json.loads(s) if isinstance(s, str) else s
                    elif "properties" in parsed or "type" in parsed:
                        schema = parsed
                    else:
                        # Maybe the whole thing is a ground_truth dict
                        schema = parsed
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Row {idx}: JSON parse error: {e}")
                continue

            if schema is None:
                continue

            issues = check_field_schema(schema)
            if issues:
                rows_with_issues += 1
                if rows_with_issues <= 20:  # Print first 20 in detail
                    print(f"Row {idx} issues:")
                    for issue in issues:
                        print(issue)
                    print()
                for issue in issues:
                    # Categorize
                    if "union type" in issue:
                        key = "union_type"
                    elif "field_schema is" in issue:
                        key = "non_dict_field_schema"
                    elif "anyOf" in issue or "oneOf" in issue or "allOf" in issue:
                        key = "anyOf/oneOf/allOf"
                    elif "$ref" in issue:
                        key = "$ref"
                    elif "items is a list" in issue:
                        key = "list_items"
                    else:
                        key = "other"
                    all_issue_types[key] = all_issue_types.get(key, 0) + 1

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total rows scanned: {total_rows}")
    print(f"Rows with issues:   {rows_with_issues} ({100*rows_with_issues/max(total_rows,1):.1f}%)")
    print(f"Issue breakdown:")
    for k, v in sorted(all_issue_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
