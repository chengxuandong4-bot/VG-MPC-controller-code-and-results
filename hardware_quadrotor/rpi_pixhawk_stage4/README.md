# F450 / TinyMPC Stage 4

Stage 4 is a fail-closed Raspberry Pi–Pixhawk real-hardware transmission framework. It is disabled by default and is separate from the frozen Stage 2 controller and Stage 3 read-only/dry-run code.

Modes:

- `audit-only` (default): no socket and no transport import.
- `bench-disarmed`: propellers removed, Pixhawk DISARMED, explicit localhost Rpanion command-link probe.
- `flight-live`: separately approved second flight only; never arms, takes off, changes mode, lands, writes parameters, or runs motor tests.

Run the offline checks:

```bash
python3 -m rpi_pixhawk_stage4.scripts.self_check_stage4
```

Read `docs/tomorrow_field_sequence.md` before any hardware work.

For Rpanion-server 0.7.0, use the bench-only dynamic-peer procedure in `docs/rpanion_v07_bench.md`; `flight-live` remains rejected for that profile.
