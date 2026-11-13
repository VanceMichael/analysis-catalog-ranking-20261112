# catalog-ranking

kind 为 exposure/click/purchase，available 是记录时刻的可售标记。

项目目前提供本地记录校验、输入指纹和按 `kind` 的原始核算。分析人员可以保留完整输入并复查每次导出的 JSON；原始数据不被命令改写。业务配置位于 `data/policy.json`，字段约定在 `docs/data.md`。

## 运行

使用 Node.js 22 LTS（>=22.22.1 且 <23）和 npm，无第三方运行依赖。
`.nvmrc` 选择 22 主版本，清单与锁文件记录相同的兼容范围；Dockerfile 仍固定基础镜像摘要，容器构建可精确复现。

```sh
npm ci --ignore-scripts --no-audit --no-fund --offline
npm run build
npm test
npm run summary
node src/cli.js inspect --input data/records.json
node src/cli.js summary --input data/records.json --out var/summary.json
npm run train
node src/cli.js recommend --model var/model.json --user u0 --k 5 --at 31
npm run demo
```

`npm run train` 按 `data/policy.json` 的时间界限切分训练/验证窗口，训练置信加权隐式 ALS 模型，
与非个性化热门基线在同一批验证用户、同一候选约束下离线比较，模型写入 `var/model.json`、
完整指标写入 `var/metrics.json`。`recommend` 加载已保存的模型，对相同请求重复复核结果一致；
未知用户与训练期无正向互动的用户走可解释的热门兜底，请求时刻不可售的商品列入 `excluded` 并说明原因。

`npm run demo` 在本地打印输入、现有核算及配置并生成 `var/summary.json`，不访问网络、不等待真实业务时间，通常几秒内完成。输出目录可删除后重新运行，测试只使用临时目录。

## 容器

```sh
docker build -t catalog-ranking:local .
docker run --rm --network none catalog-ranking:local
```

Dockerfile 固定 Node.js 基础镜像摘要，构建时执行语法检查和测试。镜像构建完成后演示无需联网；可覆盖命令运行 `npm test`。构建不需要真实账号、邮件服务、数据库或线上数据。

## 目录

- `src/dataset.js`：记录校验、规范序列化和数据指纹。
- `src/domain.js`：当前业务分组核算。
- `src/cli.js`：本地读取、检查、训练、推荐和导出入口。
- `src/output.js`：以临时文件写入后替换完整 JSON。
- `src/reco/config.js`：推荐器配置的默认值合并与严格校验。
- `src/reco/split.js`：按时间切分、可售时间线与训练期聚合。
- `src/reco/als.js`：确定性置信加权隐式 ALS 与线性求解。
- `src/reco/model.js`：模型训练、落盘校验与打分/兜底路径。
- `src/reco/evaluate.js`：离线复验指标、切片与基线比较。
- `test/`：输入约束、领域核对、推荐器与命令输出回归。
- `data/`：合成记录、独立核对值及本地业务配置。
