# Dependency License Report

**Generated:** 2026-08-25  
**Scope:** `pre-deploy` and `post-deploy` projects in the saferai monorepo

---

## Summary

| License Type | Count | Risk Level |
|---|---|---|
| MIT | 38 | Low (permissive) |
| Apache-2.0 | 22 | Low (permissive) |
| BSD-3-Clause | 16 | Low (permissive) |
| BSD-2-Clause | 3 | Low (permissive) |
| PSF-2.0 | 2 | Low (permissive) |
| MPL-2.0 | 1 | Low (weak file-level copyleft) |
| Unlicense | 1 | Low (public domain) |
| BSD-style (composite) | 1 | Low (permissive) |

**Overall assessment:** All dependencies use permissive open-source licenses (MIT, BSD, Apache-2.0) that are compatible with commercial use and with the project's own MIT license. No copyleft dependencies are present.

---

## Complete Dependency License List

### Pre-deploy Project (Direct Dependencies)

| Package | Version | License | Source |
|---|---|---|---|
| boto3 | 1.37.10 | Apache-2.0 | [PyPI](https://pypi.org/project/boto3/) |
| botocore | 1.37.10 | Apache-2.0 | [PyPI](https://pypi.org/project/botocore/) |
| datasets | 2.16.0 | Apache-2.0 | [PyPI](https://pypi.org/project/datasets/) |
| deepdiff | 8.6.1 | MIT | [PyPI](https://pypi.org/project/deepdiff/) |
| deepeval | 3.3.9 | Apache-2.0 | [PyPI](https://pypi.org/project/deepeval/) |
| nltk | 3.9.1 | Apache-2.0 | [PyPI](https://pypi.org/project/nltk/) |
| numpy | 1.26.4 | BSD-3-Clause | [PyPI](https://pypi.org/project/numpy/) |
| pyarrow | 14.0.2 | Apache-2.0 | [PyPI](https://pypi.org/project/pyarrow/) |
| pandas | 2.2.3 | BSD-3-Clause | [PyPI](https://pypi.org/project/pandas/) |
| spacy | 3.7.1 | MIT | [PyPI](https://pypi.org/project/spacy/) |

### Post-deploy Project (Direct Dependencies)

| Package | Version Spec | License | Source |
|---|---|---|---|
| pandas | >=2.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/pandas/) |
| pydantic | >=2.0 | MIT | [PyPI](https://pypi.org/project/pydantic/) |
| pyyaml | >=6.0 | MIT | [PyPI](https://pypi.org/project/PyYAML/) |
| presidio_analyzer | >=2.2 | MIT | [PyPI](https://pypi.org/project/presidio-analyzer/) |
| transformers | >=4.30 | Apache-2.0 | [PyPI](https://pypi.org/project/transformers/) |
| torch | >=2.0 | BSD-style (composite) | [PyPI](https://pypi.org/project/torch/) |
| datasets | >=2.0 | Apache-2.0 | [PyPI](https://pypi.org/project/datasets/) |
| openai | >=1.0 | Apache-2.0 | [GitHub](https://github.com/openai/openai-python) |
| boto3 | >=1.28 | Apache-2.0 | [PyPI](https://pypi.org/project/boto3/) |
| s3fs | >=2023.6 | BSD-3-Clause | [PyPI](https://pypi.org/project/s3fs/) |
| pytest | >=7.0 | MIT | [PyPI](https://pypi.org/project/pytest/) |
| ruff | >=0.1 | MIT | [PyPI](https://pypi.org/project/ruff/) |

### Transitive Dependencies (from post-deploy requirements.txt)

| Package | Version | License |
|---|---|---|
| annotated-types | 0.8.0 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.4 | MIT |
| pydantic_core | 2.46.4 | MIT |
| python-dateutil | 2.9.0.post0 | Apache-2.0 / BSD |
| six | 1.17.0 | MIT |
| spacy | 3.8.15 | MIT |
| spacy-legacy | 3.0.12 | MIT |
| spacy-loggers | 1.0.5 | MIT |
| phonenumbers | 9.0.37 | Apache-2.0 |
| regex | 2026.7.19 | Apache-2.0 |
| tldextract | 5.3.2 | BSD-3-Clause |
| requests-file | 3.0.1 | Apache-2.0 |
| tokenizers | 0.22.2 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| huggingface_hub | 1.28.0 | Apache-2.0 |
| hf-xet | 1.6.0 | Apache-2.0 |
| httpx | 0.28.1 | BSD-3-Clause |
| httpcore | 1.0.9 | BSD-3-Clause |
| anyio | 4.14.2 | MIT |
| h11 | 0.16.0 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| idna | 3.19 | BSD-3-Clause |
| Jinja2 | 3.1.6 | BSD-3-Clause |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| filelock | 3.32.3 | Unlicense |
| fsspec | 2026.6.0 | BSD-3-Clause |
| tqdm | 4.70.0 | MIT / MPL-2.0 |
| requests | 2.34.2 | Apache-2.0 |
| urllib3 | 2.7.0 | MIT |
| charset-normalizer | 3.5.1 | MIT |
| packaging | 26.3 | Apache-2.0 / BSD-2-Clause |
| click | 8.4.2 | BSD-3-Clause |
| rich | 15.0.0 | MIT |
| markdown-it-py | 4.2.0 | MIT |
| mdurl | 0.1.2 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| shellingham | 1.5.4 | MIT |
| typer | 0.27.1 | MIT |
| blis | 1.3.3 | MIT |
| catalogue | 2.0.10 | MIT |
| cymem | 2.0.13 | MIT |
| murmurhash | 1.0.15 | MIT |
| preshed | 3.0.13 | MIT |
| srsly | 2.5.3 | MIT |
| thinc | 8.3.13 | MIT |
| wasabi | 1.1.3 | MIT |
| weasel | 1.0.0 | MIT |
| smart_open | 8.0.1 | MIT |
| cloudpathlib | 0.24.0 | MIT |
| confection | 1.3.3 | MIT |
| annotated-doc | 0.0.5 | MIT |
| aiohttp | 3.14.3 | Apache-2.0 |
| aiosignal | 1.4.0 | Apache-2.0 |
| aiohappyeyeballs | 2.7.1 | PSF-2.0 |
| frozenlist | 1.8.0 | Apache-2.0 |
| multidict | 6.7.1 | Apache-2.0 |
| yarl | 1.24.5 | Apache-2.0 |
| propcache | 0.5.2 | Apache-2.0 |
| attrs | 26.1.0 | MIT |
| wrapt | 2.3.0 | BSD-2-Clause |
| pyarrow | 25.0.1 | Apache-2.0 |
| multiprocess | 0.70.19 | BSD-3-Clause |
| dill | 0.4.1 | BSD-3-Clause |
| xxhash | 4.0.1 | BSD-2-Clause |
| mpmath | 1.3.0 | BSD-3-Clause |
| sympy | 1.14.0 | BSD-3-Clause |
| networkx | 3.6.1 | BSD-3-Clause |
| iniconfig | 2.3.0 | MIT |
| pluggy | 1.6.0 | MIT |

---

## License Category Definitions

| Category | Description | Commercial Use | Distribution |
|---|---|---|---|
| **MIT** | Very permissive. Requires only copyright notice. | Yes | Yes |
| **BSD-3-Clause** | Permissive. Requires copyright notice; no endorsement clause. | Yes | Yes |
| **BSD-2-Clause** | Simplified BSD. Same as BSD-3 without the no-endorsement clause. | Yes | Yes |
| **Apache-2.0** | Permissive. Requires notice + patent grant. Compatible with GPL-3.0. | Yes | Yes |
| **PSF-2.0** | Python Software Foundation license. Permissive. | Yes | Yes |
| **MPL-2.0** | Weak copyleft at file level. Modified MPL files must be shared. | Yes | Yes (file-level copyleft) |
| **Unlicense** | Public domain dedication. No restrictions. | Yes | Yes |

---

## Recommendations

1. **certifi (MPL-2.0):** The MPL-2.0 is a file-level copyleft. It only requires sharing source for modified certifi files themselves. No action needed for normal usage.

2. **torch (composite license):** PyTorch's license combines BSD-style, Apache-2.0, MIT, and BSL-1.0 for different components. All are permissive and compatible with MIT. No action needed.

3. **All Apache-2.0 dependencies** require that you include a copy of NOTICE files if they provide one, when redistributing. This matters if the project is distributed as a package.

4. **Overall:** The dependency stack is clean for commercial use and compatible with the project's MIT license. No copyleft (GPL/LGPL) dependencies are present.

---

*This report was generated through automated research of PyPI and GitHub license metadata. For legal decisions, verify licenses against the actual LICENSE files in each package.*
