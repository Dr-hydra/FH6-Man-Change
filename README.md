# FH6 角色模型通用管线

本仓库只提供通用的 FH6 模型处理能力：流程文档、脚本、工具、测试和 Skills。具体 Mod 的源代码、角色资产、游戏文件和成品包不进入公开源码仓库。

## 工作流

1. [FBX 到 Display](docs/pipeline/fbx-to-display.md)
2. [Display 到 Driver](docs/pipeline/display-to-driver.md)
3. [Mod 打包](docs/pipeline/package-build.md)
4. [安装器与部署合同](docs/pipeline/installer.md)

脚本归属和各流程入口见 [pipeline/scripts.manifest.json](pipeline/scripts.manifest.json)。

## 仓库内容

- `scripts/`：与角色无关的模型、材质、ZIP 和校验工具
- `tools/`：可复用的命令行工具源码
- `tests/`：使用合成或脱敏夹具的测试
- `docs/`：FH6 格式、转换流程和验证规范
- `skills/`：可复用的 FH6 Skills（通过 Release 单独提供安装包）
- `pipeline/`：脚本所有权和流程约定

以下内容不进入公开源码：`mods/`、`sources/`、`releases/`、`work/`，以及 FBX、PMX、Blend、modelbin、swatchbin 和游戏原始归档。已完成的 Mod 只通过 GitHub Release 分发成品包。

用户端安装器和部署工具由独立的 [FH6Tools](https://github.com/Dr-hydra/FH6Tools) 项目维护；本仓库只定义安装器读取的 Mod 包格式、校验规则和部署合同。

## Skills

当前提供 4 个 FH6 专用 Skill：`fh6-blender-pipeline`、`fh6-head-runtime-pipeline`、`fh6-body-proportion-repair`、`fh6-character-detail-repair`。它们分别覆盖模型导入、头部运行时、身体比例修复和局部细节修复。Skill 安装包可在 Releases 页面下载。

## 许可证与署名

通用脚本、工具和测试使用 [CC BY-NC 4.0](LICENSE)；文档和 Skills 使用 [CC BY-NC 4.0](LICENSE-DOCS)。再分发时请保留作者署名、项目地址、许可证链接，并说明修改内容。禁止商业用途。

本仓库不授权 Forza Horizon 6 原版文件、提取的游戏资产或第三方角色资产；这些内容不属于本项目许可证范围，并受其各自授权条款约束。
