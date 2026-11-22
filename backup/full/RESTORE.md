# Restore 模型训练

Download all release assets, then run:

```bash
sha256sum -c SHA256SUMS
mkdir -p restored
cat ReimburseAudit-model-training.tar.zst.part-* | zstd -d | tar -xf - -C restored
```

Excluded basenames/patterns: .git, __pycache__, .pytest_cache, .cache, node_modules, .venv, venv, .ssh, .env, .env.*, .git-credentials, *.pem, *.key.
