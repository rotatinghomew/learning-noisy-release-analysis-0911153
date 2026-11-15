# 带相关发布噪声的课程参与统计及变化可信度

面向社区与老年学习项目的 Python 分析服务。当前支持 `release_inventory` 的已观察资料核对；业务资料在 `docs/data-contract.md`，现有调用与保存语义在 `docs/http-contract.md`，分析目标的验证边界在 `docs/verification.md`。

## 环境与运行

Python 3.11.14，依赖完整固定在 requirements.txt。首次准备环境安装依赖；正式运行与测试无需联网。

```sh
python -m pip install -r requirements.txt
python -m compileall -q app tests
python -m unittest discover -s tests -v
python -m app.demo
```

单个 Linux 应用容器构建与当前产品展示：

```sh
docker build -t noisy-release-analysis:initial .
docker run --rm --network none noisy-release-analysis:initial python -m unittest discover -s tests -v
docker run --rm --network none noisy-release-analysis:initial
```

`app.demo` 运行真实 HTTP 资料目录与分析进程边界并回读结果，通常数秒结束。数据全部为本地合成记录，没有外部服务或凭据；运行时不会拉包。测试与演示默认读取 data/current.json，后续可用 `--packet` 指定同契约的业务资料。所有时间及单位含义见资料契约，不以零代替缺失。

现有业务视图：`app/releases.py`，当前结果在 `source_profile` 中公开。
