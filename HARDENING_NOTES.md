# Hardening Notes

## Scope

This pass narrowed the repository to the exact archived-context runtime that actually executes today.

## 1. Public Contract

- `RFSNConfig` now exposes only real runtime controls.
- Removed stale compression-facing config knobs from the active contract.
- `RuntimeMode` is now:
  - `exact`: hot-window only, no archival spill
  - `archived`: exact hot window plus exact archived spill/restore
- `session_id` is now an explicit user-facing concept on config, CLI, and API surfaces.

## 2. Runtime Semantics

- `model.prefill()` rejects prompts that exceed `max_position_embeddings`.
- `model.generate()` rejects prompt plus requested generation that would exceed `max_position_embeddings` before execution starts.
- `model.decode_step()` validates that the caller position matches cache state and rejects decode overflow.
- `LayerKVCache.evict_for_append()` now enforces real `runtime_mode='exact'` behavior at the spill boundary.

## 3. Session-Safe Persistence

- Persisted archives are namespaced under:
  - `disk_cache_dir / model_id / session_id / layer_*`
- Restore now requires an explicit `session_id`.
- Restore distinguishes:
  - missing persisted archive for the model
  - unknown session
  - empty session
- Restore rejects unreadable or gapped archived state rather than silently continuing from a partial prefix.
- Restored caches now repair the live context cursor so appended continuation prompts resume at the correct absolute position.

## 4. Runtime Spine Cleanup

The repo no longer presents the removed experimental compression path as part of the active runtime contract.

Deleted stale modules:

- `rfsn_v10_5/codec.py`
- `rfsn_v10_5/attention_compressed.py`
- `rfsn_v10_5/fp8.py`
- `rfsn_v10_5/types.py`

Top-level exports and README examples were updated accordingly.

## 5. Tests

Updated tests cover:

- exact-mode overflow rejection
- archived-mode spill behavior
- session-scoped restore isolation
- restart continuity after restore
- prompt plus generation hard limit enforcement
- API and launcher session/restore behavior

The renamed archived-runtime regression test is now:

- `tests/test_archived_runtime.py`

## 6. Manual Verification

The following were verified by reading the runtime code, not only by running tests:

- Contract honesty:
  - `rfsn_v10_5/config.py`
  - `rfsn_v10_5/launcher.py`
  - `rfsn_v10_5/api.py`
- Execution honesty:
  - `rfsn_v10_5/model.py`
  - `rfsn_v10_5/cache.py`
  - `rfsn_v10_5/layer.py`
- Persistence safety:
  - `rfsn_v10_5/cache.py`
  - `rfsn_v10_5/storage.py`
- Limit enforcement:
  - `rfsn_v10_5/model.py`
  - `rfsn_v10_5/cache.py`

## Verification Result

- Full test suite: `python -m unittest discover -s tests`
- Result: `Ran 55 tests` / `OK`

Expected warning logs from the cold-storage integrity tests remain, because those tests intentionally exercise missing-payload, checksum-mismatch, and deserialization-failure paths.
