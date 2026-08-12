#!/bin/bash
# 修复 stepaudio2 命名冲突（stepaudio2.py 与包名冲突导致导入失败）
# 此脚本必须在 pip install stepaudio2-minicpmo 之后执行

STEPAUDIO2_PATH=$(python -c "import stepaudio2; print(stepaudio2.__path__[0])" 2>/dev/null)

if [ -z "$STEPAUDIO2_PATH" ]; then
    STEPAUDIO2_PATH=$(pip show stepaudio2 2>/dev/null | grep Location | awk '{print $2}')/stepaudio2
fi

if [ -f "$STEPAUDIO2_PATH/stepaudio2.py" ]; then
    mv "$STEPAUDIO2_PATH/stepaudio2.py" "$STEPAUDIO2_PATH/step_audio2_impl.py"
    sed -i 's/from \.stepaudio2 import/from .step_audio2_impl import/g' "$STEPAUDIO2_PATH/__init__.py"
    echo "✅ stepaudio2 命名冲突已自动修复"
else
    echo "⚠️ stepaudio2.py 不存在，可能已修复或无需修复"
fi

python -c "import stepaudio2; print('✅ stepaudio2 导入验证通过')" 2>/dev/null || echo "❌ 导入验证失败"
