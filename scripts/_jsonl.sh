# Helpers shared by train.sh / eval.sh / eval_zero_shot.sh (source, don't exec).
#
# Dataset jsonls are named by dataset, optionally followed by one extra
# dot-tag before ".jsonl" (the released MMTA files carry such a tag), e.g.
#   covla_test.jsonl            covla_test.<tag>.jsonl
#   terra_test.percha.jsonl     terra_test.percha.<tag>.jsonl
# Both forms are accepted, so data can be used as downloaded without renaming.

# Per-channel variant suffixes that are part of the dataset name, not a tag.
JSONL_VARIANT_SUFFIXES=(percha)

# resolve_jsonl DIR NAME → path of DIR/NAME.jsonl, else DIR/NAME.<tag>.jsonl
# (single dot-free tag). Prints DIR/NAME.jsonl when neither exists so the
# caller's missing-file check reports a sensible path.
resolve_jsonl() {
    local dir=$1 name=$2 f tag
    if [ -f "$dir/$name.jsonl" ]; then
        echo "$dir/$name.jsonl"
        return
    fi
    for f in "$dir/$name".*.jsonl; do
        [ -f "$f" ] || continue
        tag=${f#"$dir/$name."}
        tag=${tag%.jsonl}
        if [[ "$tag" != *.* ]] && ! _jsonl_is_variant "$tag"; then
            echo "$f"
            return
        fi
    done
    echo "$dir/$name.jsonl"
}

# jsonl_stem PATH → dataset name with ".jsonl" and any trailing tag removed:
#   covla_test.<tag>.jsonl → covla_test,  terra_test.percha.<tag>.jsonl → terra_test.percha
jsonl_stem() {
    local s
    s=$(basename "$1")
    s=${s%.jsonl}
    if [[ "$s" == *.* ]] && ! _jsonl_is_variant "${s##*.}"; then
        s=${s%.*}
    fi
    echo "$s"
}

_jsonl_is_variant() {
    local v
    for v in "${JSONL_VARIANT_SUFFIXES[@]}"; do
        [ "$1" = "$v" ] && return 0
    done
    return 1
}
