#!/bin/bash
set -x

# ============================================================
#  ai4 PyPI Publish Script
# ============================================================
#
#  Usage:
#    ./pub.sh <version> <"message"> [test|prod]
#
#    Examples:
#      ./pub.sh 0.1.0 "first release"          # dry-run
#      ./pub.sh 0.1.0 "first release" test    # Test PyPI
#      ./pub.sh 0.1.0 "first release" prod    # Production PyPI
#
#  Behavior:
#    - Writes version to pyproject.toml before building
#    - Version MUST be >= current version (enforced monotonically)
#    - Defaults to dry-run if no env specified
#
#  Prerequisites:
#    pip install twine build
#
# ============================================================

cd "$(dirname "$0")"
set -e

PKG="ai4"
TOML="pyproject.toml"

# --- Parse arguments ---
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <version> <\"message\"> [test|prod]"
    echo "Example: $0 0.1.0 \"first release\" [test|prod]"
    exit 1
fi

NEW_VERSION="$1"
COMMIT_MSG="$2"
MODE="${3:-dry-run}"

# --- Validate mode ---
case "$MODE" in
    dry-run|test|prod) ;;
    *)
        echo "Error: mode must be 'dry-run', 'test', or 'prod', got '$MODE'"
        exit 1
        ;;
esac

# --- Read current version from pyproject.toml ---
CURRENT_VERSION=$(grep '^version = ' "$TOML" | sed 's/version = "//;s/"//')

echo "=========================================="
echo "  $PKG publish check"
echo "=========================================="
echo "  Current : $CURRENT_VERSION"
echo "  New     : $NEW_VERSION"
echo "  Mode    : $MODE"
echo "=========================================="

# --- Version monotonicity check ---
# Compare versions using sort -V (version sort)
if ! printf '%s\n%s\n' "$CURRENT_VERSION" "$NEW_VERSION" | sort -V -C; then
    echo "ERROR: New version $NEW_VERSION must be >= current version $CURRENT_VERSION"
    echo "Downgrade not allowed. Aborting."
    exit 1
fi

if [ "$NEW_VERSION" = "$CURRENT_VERSION" ]; then
    echo "Note: version unchanged ($CURRENT_VERSION)"
else
    echo "Version increased: $CURRENT_VERSION -> $NEW_VERSION"
fi

# --- Write new version to pyproject.toml ---
echo ""
echo "=== Updating version in $TOML ==="
sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$TOML"
grep "^version = " "$TOML"

# --- Dry-run mode: validate only ---
if [ "$MODE" = "dry-run" ]; then
    echo ""
    echo "=== DRY-RUN: version updated, skipping build/upload ==="
    echo "Re-run with 'test' or 'prod' to publish."
    exit 0
fi

# --- Mode label for upload ---
UPLOAD_LABEL="Upload to $([ "$MODE" = "test" ] && echo 'Test PyPI' || echo 'PyPI')"

# --- 1. Run tests ---
echo ""
echo "=== 1. Running tests ==="
export PYTHONPATH="$(pwd)"
if ! uv run --no-project pytest world/tests nn/tests ml/tests -q; then
    echo "TESTS FAILED - aborting publish"
    exit 1
fi
echo "Tests passed."

# --- 2. Clean old builds ---
echo ""
echo "=== 3. Cleaning old builds ==="
rm -rf build/
rm -rf dist/
rm -rf "$PKG.egg-info/"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "Cleaned."

# --- 3. Build ---
echo ""
echo "=== 4. Building sdist + wheel ==="
python -m build
echo "Build complete:"
ls -lh dist/

# --- 4. Verify ---
echo ""
echo "=== 5. Verifying with twine check ==="
twine check dist/*
echo "Verification passed."

# --- 5. Upload ---
echo ""
echo "=== 6. $UPLOAD_LABEL ==="
if [ "$MODE" = "test" ]; then
    twine upload --repository testpypi dist/*
else
    twine upload dist/*
fi

echo ""
echo "=========================================="
echo "  $PKG v$NEW_VERSION published!"
echo "=========================================="