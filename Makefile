.PHONY: install clean test check

# 默认目标
all: install

# 安装目标
install: check-python check-pip install-deps install-package set-permissions verify-install
	@echo ""
	@echo "=========================================="
	@echo "安装完成！"
	@echo "=========================================="
	@echo "现在可以使用 'association' 命令了"
	@echo "运行 'association --help' 查看帮助信息"
	@echo "=========================================="

# 检查Python版本
check-python:
	@echo "检查Python版本..."
	@python3 --version || (echo "错误: 未找到Python3，请先安装Python 3.7或更高版本" && exit 1)
	@python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" || \
		(echo "错误: Python版本过低，需要Python 3.7或更高版本" && exit 1)
	@echo "Python版本检查通过"

# 检查pip
check-pip:
	@echo "检查pip..."
	@python3 -m pip --version || (echo "错误: 未找到pip，请先安装pip" && exit 1)
	@echo "pip检查通过"

# 安装Python依赖
install-deps:
	@echo "安装Python依赖包..."
	@python3 -m pip install --upgrade pip -q
	@python3 -m pip install -r requirements.txt
	@echo "依赖包安装完成"

# 安装项目本身
install-package:
	@echo "安装assoG2P包..."
	@python3 -m pip install -e .
	@echo "包安装完成"

# 设置软件可执行文件权限
set-permissions:
	@echo "设置软件可执行文件权限..."
	@chmod +x assoG2P/bin/software/plink 2>/dev/null || true
	@chmod +x assoG2P/bin/software/gemma-0.98.5-linux-static-AMD64 2>/dev/null || true
	@echo "权限设置完成"

# 验证安装
verify-install:
	@echo "验证安装..."
	@which association > /dev/null && echo "✓ association命令已安装" || \
		(echo "✗ 警告: association命令未找到，请检查PATH环境变量" && exit 1)
	@association --help > /dev/null 2>&1 && echo "✓ association命令可以正常运行" || \
		(echo "✗ 警告: association命令无法运行" && exit 1)
	@echo "安装验证完成"

# 测试安装（可选）
test:
	@echo "运行测试..."
	@association --help
	@echo "测试完成"

# 清理安装文件
clean:
	@echo "清理安装文件..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf __pycache__/
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "清理完成"

# 卸载（可选）
uninstall:
	@echo "卸载assoG2P..."
	@python3 -m pip uninstall -y assoG2P 2>/dev/null || true
	@echo "卸载完成"

# 显示帮助信息
help:
	@echo "Makefile 使用说明:"
	@echo ""
	@echo "  make          - 安装项目（默认）"
	@echo "  make install  - 安装项目"
	@echo "  make test     - 测试安装"
	@echo "  make clean    - 清理临时文件"
	@echo "  make uninstall - 卸载项目"
	@echo "  make help     - 显示此帮助信息"
	@echo ""
