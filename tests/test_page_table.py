from __future__ import annotations

import unittest

from rfsn_v10_5.block_manager import (
    BlockId,
    BlockLocation,
    BlockManager,
    BlockManifest,
    BlockSpan,
    PageTable,
)


class PageTableTest(unittest.TestCase):
    def _manifest(
        self,
        block_name: str,
        start: int,
        end: int,
        location: BlockLocation,
        *,
        layer_id: int = 0,
        materializable: bool = True,
    ) -> BlockManifest:
        return BlockManifest(
            block_id=BlockId("model-a", layer_id, block_name),
            span=BlockSpan(start, end),
            codec_version="v11-exact-kv",
            dtype="float32",
            shape_metadata={
                "keys": (1, 2, end - start, 4),
                "values": (1, 2, end - start, 4),
            },
            checksum=f"checksum-{block_name}",
            residency=location,
            materializable=materializable,
        )

    def test_page_table_tracks_ranges_and_residency_transitions(self) -> None:
        manager = BlockManager("model-a")
        manifests = [
            self._manifest("blk-cold", 8, 12, BlockLocation.COLD_DISK),
            self._manifest("blk-hot", 0, 4, BlockLocation.HOT),
            self._manifest("blk-missing", 12, 16, BlockLocation.MISSING, materializable=False),
            self._manifest("blk-warm", 4, 8, BlockLocation.WARM_RAM),
        ]

        for manifest in manifests:
            manager.register_block(manifest)

        located = manager.locate_blocks_for_range(0, 2, 14)
        self.assertEqual(
            [manifest.block_id.block_id for manifest in located],
            ["blk-hot", "blk-warm", "blk-cold", "blk-missing"],
        )

        manager.demote_block(manifests[1].block_id, BlockLocation.WARM_RAM)
        manager.promote_block(manifests[0].block_id, BlockLocation.HOT)
        manager.mark_missing(manifests[2].block_id, reason="payload missing")
        manager.mark_unmaterializable(
            layer_id=0,
            span=BlockSpan(8, 12),
            reason="corrupt payload quarantined",
        )

        self.assertEqual(manager.get_block(manifests[1].block_id).residency, BlockLocation.WARM_RAM)
        self.assertEqual(manager.get_block(manifests[0].block_id).residency, BlockLocation.HOT)
        self.assertFalse(manager.get_block(manifests[0].block_id).materializable)
        self.assertEqual(
            manager.get_block(manifests[2].block_id).failure_reason,
            "payload missing",
        )

        stats = manager.get_residency_stats()
        self.assertEqual(stats["total_blocks"], 4)
        self.assertEqual(stats["total_tokens"], 16)
        self.assertEqual(stats["by_location"][BlockLocation.HOT.value]["blocks"], 1)
        self.assertEqual(stats["by_location"][BlockLocation.WARM_RAM.value]["blocks"], 2)
        self.assertEqual(stats["by_location"][BlockLocation.MISSING.value]["blocks"], 1)
        self.assertEqual(stats["unmaterializable_blocks"], 2)

    def test_register_rejects_overlapping_ranges(self) -> None:
        manager = BlockManager("model-a")
        manager.register_block(self._manifest("blk-a", 0, 4, BlockLocation.HOT))

        with self.assertRaisesRegex(ValueError, "overlaps"):
            manager.register_block(self._manifest("blk-b", 3, 6, BlockLocation.COLD_DISK))

    def test_metadata_round_trip_rebuilds_manager_and_page_table(self) -> None:
        manager = BlockManager("model-a")
        manager.register_block(self._manifest("blk-layer0-a", 0, 4, BlockLocation.WARM_RAM))
        manager.register_block(self._manifest("blk-layer1-a", 0, 8, BlockLocation.COLD_DISK, layer_id=1))

        serialized = manager.serialize_metadata()
        restored_manager = BlockManager.deserialize_metadata(serialized)
        restored_table = PageTable.from_dict(serialized["page_table"])

        restored_layer0 = restored_manager.locate_blocks_for_range(0, 0, 4)
        restored_layer1 = restored_manager.locate_blocks_for_range(1, 0, 8)
        table_layer1 = restored_table.locate(1, 0, 8)

        self.assertEqual(restored_layer0[0].block_id.block_id, "blk-layer0-a")
        self.assertEqual(restored_layer1[0].block_id.block_id, "blk-layer1-a")
        self.assertEqual(table_layer1[0].checksum, "checksum-blk-layer1-a")


if __name__ == "__main__":
    unittest.main()