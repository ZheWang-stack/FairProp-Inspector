# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Examples directory with quickstart and edge inference demos
- Performance comparison table in README
- ROADMAP.md showing project vision
- CHANGELOG.md for tracking changes
- Issue templates for bug reports and feature requests

### Changed
- Updated installation instructions to use git instead of PyPI
- Fixed script paths in documentation (scripts/ → src/generator/)
- Fixed parameter names in examples (--num_samples → --count)
- Improved GitHub Pages deployment to only include static files

### Fixed
- CODE_OF_CONDUCT.md contact email placeholder
- Documentation inconsistencies between README and actual code

## [0.1.0] - 2026-01-20

### Added
- Initial release
- ModernBERT-based FHA compliance classifier
- Training pipeline with Flash Attention support
- ONNX export for edge deployment
- Synthetic data generation framework
- Basic CLI interface
- Landing page with interactive terminal demo

### Known Issues
- PyPI package not yet published (install from source)
- Limited test coverage
- English-only support

[Unreleased]: https://github.com/ZheWang-stack/FairProp-Inspector/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZheWang-stack/FairProp-Inspector/releases/tag/v0.1.0
