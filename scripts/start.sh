#!/bin/bash
#
# EdgeRunner 启动脚本
# 用法: ~/start_edgerunner.sh
#
# 创建一个 tmux session，布局如下:
# ┌─────────────────────────────────────────────────────────────────┐
# │                       Pane 0 (TOP - Long)                        │
# │                    [交互式 shell] (光标在这里)                    │
# ├─────────────────────────────────┬───────────────────────────────┤
# │       Pane 1 (Bottom Left)       │      Pane 2 (Bottom Right)    │
# │       python -m src.main         │           jtop                │
# │                                  │                               │
# └─────────────────────────────────┴───────────────────────────────┘

set -e

SESSION_NAME="edgerunner"
PROJECT_DIR="$HOME/Projects/edge_runner"
VENV_PATH="$HOME/vlm_env/bin/activate"

echo "🚀 EdgeRunner 启动脚本"
echo "========================"

# ============================================================
# Phase 1: 系统准备
# ============================================================
echo "⚡ Step 1: 运行 jetson_clocks..."
sudo jetson_clocks
echo "   ✅ jetson_clocks 已启用"

echo "🔇 Step 2: 抑制内核消息..."
sudo dmesg -n 1
echo "   ✅ dmesg 已静默"

# ============================================================
# Phase 2: 清理旧 session
# ============================================================
echo "🧹 Step 3: 检查旧 tmux session..."
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "   发现旧 session，正在清理..."
    tmux kill-session -t "$SESSION_NAME"
    echo "   ✅ 旧 session 已清理"
else
    echo "   ✅ 无旧 session"
fi

# ============================================================
# Phase 3: 等待 llama-server 就绪
# ============================================================
echo "🧠 Step 4: 等待 llama-server 服务..."
MAX_WAIT=30
WAITED=0
while ! curl -s http://localhost:8080/health > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "   ⚠️  llama-server 未就绪，但继续启动..."
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    echo -n "."
done
if [ $WAITED -lt $MAX_WAIT ]; then
    echo ""
    echo "   ✅ llama-server 就绪"
fi

# ============================================================
# Phase 4: 创建 tmux session 和布局
# ============================================================
echo "📺 Step 5: 创建 tmux session..."

# 创建新 session（detached 模式），工作目录为项目目录
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR"

# 垂直分割：创建上下两个 pane (pane 0 在上, pane 1 在下)
tmux split-window -v -t "$SESSION_NAME:0.0" -c "$PROJECT_DIR"

# 水平分割底部 pane：创建左右两个 pane (pane 1 左下, pane 2 右下)
tmux split-window -h -t "$SESSION_NAME:0.1" -c "$PROJECT_DIR"

# 调整顶部 pane 大小（占 65%）
tmux resize-pane -t "$SESSION_NAME:0.0" -y 65%

echo "   ✅ 布局创建完成"

# ============================================================
# Phase 5: 向每个 pane 发送命令
# ============================================================
echo "🔧 Step 6: 配置各 pane..."

# Pane 0 (TOP): 激活环境，保持交互式 (光标停在这里)
tmux send-keys -t "$SESSION_NAME:0.0" "source $VENV_PATH" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "# 🎯 EdgeRunner 控制台 - 可在此执行命令" C-m

# Pane 1 (Bottom Left): 激活环境 + 运行 main.py
tmux send-keys -t "$SESSION_NAME:0.1" "source $VENV_PATH && python -m src.main" C-m

# Pane 2 (Bottom Right): 运行 jtop
tmux send-keys -t "$SESSION_NAME:0.2" "jtop" C-m

echo "   ✅ 命令已发送"

# ============================================================
# Phase 6: 选择活动 pane 并 attach
# ============================================================
echo "🎯 Step 7: 切换到控制台 pane..."

# 选择 Pane 0 (TOP) 作为活动 pane
tmux select-pane -t "$SESSION_NAME:0.0"

echo ""
echo "========================"
echo "✅ EdgeRunner 启动完成!"
echo "========================"
echo ""
echo "📝 布局说明:"
echo "   • 顶部: 交互式 shell (当前光标位置)"
echo "   • 左下: python -m src.main (主程序)"
echo "   • 右下: jtop (系统监控)"
echo ""
echo "🎮 Tmux 快捷键:"
echo "   • Ctrl+b ↑/↓/←/→  切换 pane"
echo "   • Ctrl+b d        detach (后台运行)"
echo "   • Ctrl+b z        zoom 当前 pane"
echo ""
echo "正在 attach 到 session..."
sleep 1

# Attach 到 session
tmux attach -t "$SESSION_NAME"
