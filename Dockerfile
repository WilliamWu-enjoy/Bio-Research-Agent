name: CI

# 每次提交和 PR 自动触发
on:
  push:
    branches: [ "main", "dev" ]
  pull_request:
    branches: [ "main", "dev" ]

jobs:
  basic-checks:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: 'pip' # 自动缓存依赖，加速后续构建

      - name: Install dependencies
        # 指定在 Bio_Agent 目录下执行命令
        working-directory: ./Bio_Agent
        run: |
          python -m pip install --upgrade pip
          # 安装项目依赖，并额外安装测试所需的 httpx (FastAPI TestClient 依赖它)
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install httpx pytest

      - name: Syntax check (py_compile)
        # 自动遍历 Bio_Agent 目录下所有的 python 文件进行语法校验
        working-directory: ./Bio_Agent
        run: |
          python -m compileall .

      - name: Import smoke test
        # 检查关键模块不会导入即崩
        working-directory: ./Bio_Agent
        env:
          OPENAI_API_KEY: "ci_dummy_key"
          OPENAI_BASE_URL: "https://api.openai.com/v1"
          APP_API_KEY: "ci_dummy_app_key"
        run: |
          python -c "import config; print('config_import_ok')"
          python -c "import factory; print('factory_import_ok')"
          python -c "import api_app; print('api_app_import_ok')"

      - name: FastAPI route smoke test
        # 验证健康接口和鉴权逻辑仍有效
        working-directory: ./Bio_Agent
        env:
          OPENAI_API_KEY: "ci_dummy_key"
          OPENAI_BASE_URL: "https://api.openai.com/v1"
          APP_API_KEY: "ci_dummy_app_key"
          # 如果你的代码里有从环境变量读取配置，确保这些占位符存在
        run: |
          python - << 'PY'
          from fastapi.testclient import TestClient
          import sys

          # 尝试导入 app
          try:
              from api_app import app
          except ImportError as e:
              print(f"Import Error: {e}")
              sys.exit(1)

          client = TestClient(app)

          # 1. health 白名单接口：应可访问
          try:
              r = client.get("/v1/health")
              print(f"Health check status: {r.status_code}")
              assert r.status_code == 200
              body = r.json()
              assert "code" in body
          except Exception as e:
              print(f"Health check failed: {e}")
              sys.exit(1)

          # 2. chat 不带 key：验证鉴权逻辑是否触发 (应返回 400 或 401，符合你的业务逻辑 1002 错误码)
          try:
              r2 = client.post("/v1/chat", json={"query": "test", "chat_history": []})
              body2 = r2.json()
              print(f"Auth check response code: {body2.get('code')}")
              # 根据你描述的业务逻辑，鉴权失败 code 为 1002
              assert body2.get("code") == 1002
          except Exception as e:
              print(f"Auth check logic failed: {e}")
              sys.exit(1)

          print("Route smoke test successfully passed!")
          PY