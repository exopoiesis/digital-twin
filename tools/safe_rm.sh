#!/bin/bash
# safe_rm: замена rm -rf для рабочих директорий
# Использование: source /workspace/safe_rm.sh; safe_rm /workspace/old_dir
#
# Что делает:
# 1. Проверяет fuser — есть ли процессы с открытыми файлами
# 2. Если есть — ОТКАЗЫВАЕТСЯ удалять, показывает PID
# 3. Если нет — переносит в /workspace/.trash/name_TIMESTAMP
# 4. safe_rm_purge — удаляет trash старше 24ч

TRASH_DIR="/workspace/.trash"

safe_rm() {
    local target="$1"
    if [ -z "$target" ]; then
        echo "safe_rm: usage: safe_rm /path/to/dir_or_file" >&2
        return 1
    fi

    if [ ! -e "$target" ]; then
        echo "safe_rm: $target does not exist, nothing to do"
        return 0
    fi

    # Check if any process has open files in this path
    if command -v fuser >/dev/null 2>&1; then
        local pids
        pids=$(fuser -s "$target" 2>/dev/null; fuser -s "$target"/* 2>/dev/null)
        if fuser "$target" "$target"/* 2>/dev/null | grep -q '[0-9]'; then
            echo "safe_rm: BLOCKED — processes have open files in $target:" >&2
            fuser -v "$target" "$target"/* 2>&1 | head -10 >&2
            echo "safe_rm: use 'rm -rf $target' manually if you're sure" >&2
            return 1
        fi
    fi

    # Move to trash
    mkdir -p "$TRASH_DIR"
    local basename=$(basename "$target")
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local trash_path="$TRASH_DIR/${basename}_${timestamp}"

    mv "$target" "$trash_path" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "safe_rm: moved $target -> $trash_path"
    else
        echo "safe_rm: mv failed for $target (immutable? permissions?)" >&2
        return 1
    fi
}

# Purge trash older than 24 hours
safe_rm_purge() {
    if [ -d "$TRASH_DIR" ]; then
        local count=$(find "$TRASH_DIR" -maxdepth 1 -mmin +1440 | wc -l)
        find "$TRASH_DIR" -maxdepth 1 -mmin +1440 -exec rm -rf {} +
        echo "safe_rm_purge: removed $count items older than 24h from $TRASH_DIR"
    fi
}

# Show what's in trash
safe_rm_ls() {
    if [ -d "$TRASH_DIR" ]; then
        echo "=== Trash ($TRASH_DIR) ==="
        ls -lh --time-style=long-iso "$TRASH_DIR/" 2>/dev/null
        echo "Total: $(du -sh "$TRASH_DIR" 2>/dev/null | cut -f1)"
    else
        echo "Trash is empty"
    fi
}
