# Backup and restoration

## Current delivery: ready

The [essential-backup-20261120 release](https://github.com/zhoudapig13/ReimburseAudit/releases/tag/essential-backup-20261120) contains the current delivery directory, including its trained model heads, reader LoRA, data, source, patent documents, review documents, and evaluation records. Its three data volumes total 616,932,683 compressed bytes. All volumes and metadata were checked against GitHub's SHA-256 digests.

This snapshot excludes `模型工程存档_20261027_144223/` (the historical training archive), disposable environments/caches, Git metadata, and credential-file patterns. Exact exclusions and volume checksums are recorded in [essential/BACKUP_COMPLETE.json](essential/BACKUP_COMPLETE.json).

```bash
gh release download essential-backup-20261120 --repo zhoudapig13/ReimburseAudit --dir current-delivery
cd current-delivery
sha256sum -c SHA256SUMS
mkdir -p restored
cat ReimburseAudit-essential.tar.zst.part-* | zstd -d | tar -xf - -C restored
```

The restored project is `restored/模型训练/`. Zstandard and GitHub CLI must be installed to run these commands.

## Complete historical archive: separate upload

The [server-backup-20261120 release](https://github.com/zhoudapig13/ReimburseAudit/releases/tag/server-backup-20261120) is the separate full-directory upload, including the historical training archive. It is complete only when its own `BACKUP_COMPLETE.json` and `SHA256SUMS` are present. Download it into a different directory; do not mix volumes from different releases.

The original source inventory is recorded in [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json). Local originals have not been deleted.
