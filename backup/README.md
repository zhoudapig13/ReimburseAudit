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

## Complete historical archive: uploaded and verified

The [server-backup-20261120 release](https://github.com/zhoudapig13/ReimburseAudit/releases/tag/server-backup-20261120) contains the full-directory snapshot, including the historical training archive. All 57 data volumes (15,276,429,835 compressed bytes) and the completion manifest have been verified against GitHub SHA-256 digests. The exact archive exclusions and checksums are in [full/BACKUP_COMPLETE.json](full/BACKUP_COMPLETE.json).

Download this archive into a separate directory; do not mix volumes from different releases.

```bash
gh release download server-backup-20261120 --repo zhoudapig13/ReimburseAudit --dir full-backup
cd full-backup
sha256sum -c SHA256SUMS
mkdir -p restored
cat ReimburseAudit-model-training.tar.zst.part-* | zstd -d | tar -xf - -C restored
```

The original source inventory is recorded in [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json). Local originals have not been deleted.
